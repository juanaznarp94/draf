import threading
import queue
import copy
from typing import Union, Optional, List, Tuple, Dict
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from federated_ex.logger import Logger, ServerMetrics, flush_watchtower

from federated_ex.drift import DriftDetection


class FedPenAvg(FedAvg):
    def __init__(
            self,
            metagg,
            device: str,
            logger_instance: Logger,
            strategy_type: str,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.metagg = metagg
        self.device = device
        self.prev_parameters: Optional[Parameters] = None
        self.logger_instance = logger_instance
        self.strategy_type = strategy_type
        print(f"[Init] FedPenAvg initialized with strategy: {self.strategy_type}, device: {self.device}")

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[BaseException, Tuple[ClientProxy, FitRes]]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            print(f"\n[Round {server_round}] Some client failed {failures}")
            return None, {}

        print(f"\n[Round {server_round}] Starting aggregation. Strategy: {self.strategy_type}")
        print(f"[Round {server_round}] Number of clients: {len(results)}, Failures: {len(failures)}")
        parameters_aggregated = None
        metrics_aggregated = {}

        if self.strategy_type == "custom" and self.prev_parameters:
            print(f"[Round {server_round}] Applying drift correction")
            parameters_aggregated = self._drift_correction(server_round, results)
        else:
            for client, fit_res in results:
                partition_id = int(fit_res.metrics.get("partition_id", -1))
                print(f"[Round {server_round}] Logging client {partition_id} metrics")
                self.logger_instance.addServerMetric(ServerMetrics(
                    round=server_round,
                    client_id=partition_id,
                    dataDrift=0.0,
                    conceptDrift=0.0,
                    penality=0.0,
                    strategy=self.strategy_type,
                    drift=fit_res.metrics.get("drift", "UNKNOWN"),
                    strongness=fit_res.metrics.get("strongness", "WEAK"),
                    client_loss=fit_res.metrics.get("train_loss", 0.0),
                ))
            print(f"[Round {server_round}] Skipping drift correction")
            print(f"[Round {server_round}] Aggregating client updates")
            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                server_round, results, failures
            )

        print(f"[Round {server_round}] Performing centralized evaluation")
        evaluated_loss, evaluated_metrics = self.evaluate_fn(
            server_round, parameters_to_ndarrays(parameters_aggregated), {}
        )
        global_loss = evaluated_loss
        global_accuracy = evaluated_metrics.get("centralized_accuracy", 0.0)
        print(f"[Round {server_round}] Centralized loss: {global_loss}, accuracy: {global_accuracy}")

        self.logger_instance.add_globalMetrics(global_loss, global_accuracy)
        self.logger_instance.fire()

        print(f"[Round {server_round}] Updating previous parameters")
        self.prev_parameters = parameters_aggregated

        return parameters_aggregated, metrics_aggregated

    def _drift_correction(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
    ) -> Parameters:
        print(f"[Round {server_round}] Starting drift correction for {len(results)} clients")

        drift_detector = DriftDetection(
            metaAggregator=self.metagg,
            strategy="custom",
            server_round=server_round,
            global_parameters=self.prev_parameters,
            client_parameters=results)

        res, metrics = drift_detector.detectDrift()

        for metric in metrics:
            self.logger_instance.addServerMetric(metric)
        print(f"[Round {server_round}] Drift correction completed")
        return res
