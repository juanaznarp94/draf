import logging
from flwr.common import Parameters, parameters_to_ndarrays, ndarrays_to_parameters, NDArrays, FitRes
from flwr.server.client_proxy import ClientProxy
import numpy as np
from torchvision.models import ResNet
import torch
from torch import nn, optim
from typing import List, Tuple
from .task import model, set_weights, predict, validate_model, get_weights, predict_probs, filter_correct_predictions
from federated_ex.logger import ServerMetrics
from federated_ex.setup import loadData

import torch
import torch.nn as nn
from .metaAggr import server_aggregate_round
from .metaAggr import MetaAggregator


def psi_test(expected_probs, actual_probs, epsilon=1e-6):
    expected_probs = np.clip(expected_probs, epsilon, 1 - epsilon)
    actual_probs = np.clip(actual_probs, epsilon, 1 - epsilon)
    psi_val = np.sum((expected_probs - actual_probs) * np.log(expected_probs / actual_probs), axis=1)
    return np.mean(psi_val)


class DriftDetection:
    def __init__(self, metaAggregator: MetaAggregator, global_parameters: Parameters,
                 client_parameters: List[Tuple[ClientProxy, FitRes]], server_round: int, strategy: str):
        if server_round < 1:
            raise ValueError("Drift detection only works after at least 1 round")

        logging.error(f"[Init] Initializing drift detection for round {server_round}")
        self.global_parameters = global_parameters
        self.client_parameters = client_parameters
        self.server_round = server_round
        self.strategy = strategy
        self.clientModels = {}
        self.amountClients = len(client_parameters)

        self.metaAggregator = metaAggregator
        self.global_model: ResNet = model()
        set_weights(self.global_model, parameters_to_ndarrays(global_parameters))
        self.global_model.eval()
        self.train, self.test, _ = loadData(0, server_round, 32)

        logging.error("[Init] Initialization complete")

    def psi(self, pred1, pred2):
        logging.error("[PSI] Calculating Population Stability Index")
        try:
            psi_res = psi_test(pred2, pred1)
            return 1 - np.exp(-psi_res)
        except Exception as e:
            logging.error(f"[PSI][ERROR] {e}")
            return 0.0

    def _add_properties_to_client_models(self, client_id, key, value):
        if client_id not in self.clientModels:
            self.clientModels[client_id] = {}
        self.clientModels[client_id][key] = value

    def _detect_data_drift(self, client_id, client_model: ResNet):
        logging.info(f"[Client {client_id}][DataDrift] Detecting data drift")
        try:
            true_labels = filter_correct_predictions(self.global_model, self.train, "cpu")
            pred_client = predict_probs(client_model, true_labels, "cpu")
            pred_global = predict_probs(self.global_model, true_labels, "cpu")
            data_drift = self.psi(pred_client, pred_global)
            logging.info(f"[Client {client_id}][DataDrift] Data drift = {data_drift:.4f}")
            self._add_properties_to_client_models(client_id, "dataDrift", data_drift)
        except Exception as e:
            logging.error(f"[Client {client_id}][DataDrift][ERROR] {e}")
            self._add_properties_to_client_models(client_id, "dataDrift", 0.0)

    def _detect_concept_drift(self, client_id, client_model: ResNet):
        logging.info(f"[Client {client_id}][ConceptDrift] Detecting concept drift")
        try:
            true_labels = filter_correct_predictions(self.global_model, self.train, "cpu")
            client_preds = predict(client_model, true_labels, "cpu")
            global_preds = predict(self.global_model, true_labels, "cpu")
            concept_drift = sum(p1 != p2 for p1, p2 in zip(client_preds, global_preds)) / len(client_preds)
            logging.info(f"[Client {client_id}][ConceptDrift] Concept drift = {concept_drift:.4f}")
            self._add_properties_to_client_models(client_id, "conceptDrift", concept_drift)
        except Exception as e:
            logging.error(f"[Client {client_id}][ConceptDrift][ERROR] {e}")
            self._add_properties_to_client_models(client_id, "conceptDrift", 0.0)

    def detectDrift(self) -> tuple[Parameters, List[ServerMetrics]]:
        logging.info("[DetectDrift] Starting drift detection")
        updated_results = []
        for idx, (client, fit_res) in enumerate(self.client_parameters):
            client_id = int(fit_res.metrics.get("partition_id", -1))
            if client_id == 0:
                continue
            logging.info(f"[Client {client_id}] Processing client data")
            client_model = model()
            set_weights(client_model, parameters_to_ndarrays(fit_res.parameters))
            client_model.eval()

            self._add_properties_to_client_models(client_id, key="model", value=client_model)
            self._add_properties_to_client_models(client_id, "clientProxy", client)
            self._add_properties_to_client_models(client_id, "params", fit_res)
            self._add_properties_to_client_models(client_id, "drift", fit_res.metrics.get("drift", "UNKNOWN"))
            self._add_properties_to_client_models(client_id, "strongness", fit_res.metrics.get("strongness", "UNKNOWN"))
            self._add_properties_to_client_models(client_id, "client_loss", fit_res.metrics.get("train_loss", 0))
            self._add_properties_to_client_models(client_id, "num_examples", fit_res.num_examples)
            self._add_properties_to_client_models(client_id, "client_params", fit_res.parameters)
            self._detect_data_drift(client_id, client_model)
            self._detect_concept_drift(client_id, client_model)

        params = server_aggregate_round(
            global_model=self.global_model,
            client_weights_dict={client_id: cl["client_params"] for (client_id, cl) in self.clientModels.items()},
            client_samples_dict={client_id: cl["num_examples"] for (client_id, cl) in self.clientModels.items()},
            client_losses_dict={client_id: cl["client_loss"] for (client_id, cl) in self.clientModels.items()},
            dataDriftDict={client_id: cl["dataDrift"] for (client_id, cl) in self.clientModels.items()},
            conceptDriftDict={client_id: cl["conceptDrift"] for (client_id, cl) in self.clientModels.items()},
            meta_aggregator=self.metaAggregator

        )

        server_metrics = []
        for i, (client_id, cl) in enumerate(self.clientModels.items()):
            server_metric = ServerMetrics(
                client_id=client_id,
                round=self.server_round,
                drift=cl["drift"],
                strongness=cl["strongness"],
                dataDrift=cl["dataDrift"],
                conceptDrift=cl["conceptDrift"],
                penality=0.0,
                strategy=self.strategy,
                client_loss=cl["client_loss"]
            )
            server_metrics.append(server_metric)

        logging.error("[DetectDrift] Drift detection complete")
        return params, server_metrics
