# src/utils/lightning_launcher.py
import logging
from typing import Any, List, Optional
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, OmegaConf
from lightning_sdk import Studio, Job, Machine

log = logging.getLogger(__name__)

class LightningLauncher(Launcher):
    def __init__(
        self,
        studio_name: str,
        teamspace: str,
        user: str,
        machine: str = "A10G",
        setup_command: Optional[str] = None,
    ) -> None:
        self.studio_name = studio_name
        self.teamspace = teamspace
        self.user = user
        self.machine = machine
        self.setup_command = setup_command

    def setup(self, *, hydra_context: HydraContext, task_function: TaskFunction, config: DictConfig) -> None:
        self.hydra_context = hydra_context
        self.task_function = task_function
        self.config = config

    def launch(self, job_overrides: List[List[str]], initial_job_idx: int) -> List[Any]:
        # Initialize and start the Studio environment
        studio = Studio(name=self.studio_name, teamspace=self.teamspace, user=self.user)
        log.info(f"Connecting to Lightning Studio: {self.studio_name}")
        studio.start()

        results = []
        for idx, overrides in enumerate(job_overrides):
            job_idx = initial_job_idx + idx
            arg_str = " ".join(overrides)
            
            # Construct the execution command
            command = f"uv run src/train.py {arg_str}"
            if self.setup_command:
                command = f"{self.setup_command} && {command}"

            log.info(f"Launching Job {job_idx}: {command}")
            
            # Submit the job to Lightning AI
            job = Job.run(
                command=command,
                name=f"hydra-job-{job_idx}",
                machine=getattr(Machine, self.machine, Machine.A10G),
                studio=studio
            )
            results.append(job)

        return results
