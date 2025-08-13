from typing import Any, Optional
import json
import boto3
import os
import time
import logging
import datetime
import watchtower


class ServerMetrics:
    round: int
    global_loss: float
    global_accuracy: float
    client_id: int
    drift: str
    strongness: str
    strategy: str
    dataDrift: str
    conceptDrift: str
    penality: str
    client_loss: str

    def __init__(self, round: int, client_id, dataDrift, conceptDrift, drift, strongness, strategy, penality,
                 global_loss=None, global_accuracy=None, client_loss=None):
        self.strongness = strongness
        self.round = round
        self.drift = drift
        self.client_id = client_id
        self.global_loss = global_loss
        self.global_accuracy = global_accuracy
        self.strategy = strategy
        self.conceptDrift = conceptDrift
        self.dataDrift = dataDrift
        self.penality = penality
        self.client_loss = client_loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round,
            "global_loss": self.global_loss,
            "global_accuracy": self.global_accuracy,
            "client_id": self.client_id,
            "drift": self.drift,
            "strongness": self.strongness,
            "conceptDrift": self.conceptDrift,
            "dataDrift": self.dataDrift,
            "penality": self.penality,
            "client_loss": self.client_loss
        }


class Logger:
    def __init__(self, run_id: str, region_name: str,
                 log_group: str = "/flower/server-metrics",
                 aws_access_key: str = None,
                 aws_secret_key: str = None,
                 aws_session_token: str = None):

        safe_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        stream_name = f"global-metrics-run-{run_id}-{safe_timestamp}"
        self.logger = logging.getLogger(f"server-metrics-{run_id}")
        self.logger.setLevel(logging.INFO)
        self.metrics: list[ServerMetrics] = []
        if not any(isinstance(h, watchtower.CloudWatchLogHandler) for h in self.logger.handlers):
            handler = watchtower.CloudWatchLogHandler(
                log_group_name=log_group,
                log_stream_name=stream_name,
                boto3_client=boto3.client("logs",
                                          aws_access_key_id=aws_access_key,
                                          aws_secret_access_key=aws_secret_key,
                                          aws_session_token=aws_session_token,
                                          region_name=region_name
                                          ),
                use_queues=False

            )
            self.logger.addHandler(handler)

    def log_metrics(self, metric: dict):
        self.logger.info(metric)

    def addServerMetric(self, server_metric: ServerMetrics):
        self.metrics.append(server_metric)

    def add_globalMetrics(self, global_loss, global_accuracy):
        for metric in self.metrics:
            metric.global_accuracy = global_accuracy
            metric.global_loss = global_loss

    def fire(self):
        for server_metric in self.metrics:
            metrics_dict = server_metric.to_dict()

            self.log_metrics(
                metrics_dict,
            )

        self.metrics.clear()


def flush_watchtower(logger):
    for handler in logger.handlers:
        if isinstance(handler, watchtower.CloudWatchLogHandler):
            handler.flush()
            handler.close()
