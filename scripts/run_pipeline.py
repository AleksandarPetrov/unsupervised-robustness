#!/usr/bin/env python3

import argparse
import os
import yaml
import copy
from typing import List, Dict
from colorama import init
from termcolor import colored
from enum import Enum
import multiprocessing
import datetime
import time
import subprocess
from hashlib import sha256
import re
import logging
import shutil
from collections.abc import MutableMapping
import fnmatch

from rich.console import Console
from rich.columns import Columns
from rich.text import Text
from rich.panel import Panel

init()

parser = argparse.ArgumentParser(description="Run pipelines.")
parser.add_argument("pipeline", help="Which pipeline to run.")
parser.add_argument(
    "--cpus",
    dest="cpus",
    default=os.cpu_count(),
    type=int,
    help="Maximum number of cpus allowed to use",
)
parser.add_argument(
    "--gpus",
    dest="gpus",
    default="",
    help='CUDA devices to use, separated by commas, e.g. "2,4,5".',
)
parser.add_argument(
    "--dry-run",
    dest="dry_run",
    default="",
    action="store_true",
    help="Just show the current status without launching anything.",
)
parser.add_argument(
    "--single-tasks",
    dest="single_tasks",
    default=None,
    help="Select a single task to run.",
)
parser.add_argument(
    "--just-show",
    dest="just_show",
    action="store_true",
    help="Don't run anything, just show the commands to be run.",
)
args = parser.parse_args()

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logs_dir = f"{args.pipeline}_data_/logs/{experiment_timestamp}"
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    filename=f"{logs_dir}/log.txt",
    level=logging.DEBUG,
    format="%(asctime)s [ %(funcName)15s() ] %(message)s",
)

ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def worker(command, returncode, log_name, cuda_devices=""):
    try:
        command.strip()
        os.makedirs(os.path.dirname(log_name), exist_ok=True)
        output = open(log_name, "w")
        env = {}
        env.update(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": cuda_devices})
        # command = f"echo '{command}'; sleep 2"
        output.write(f"COMMAND: {command}\n\n")
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, shell=True, env=env)
        output.close()
        returncode.value = result.returncode
    except Exception as e:
        returncode.value = 1
        print(f"EXCEPTION {str(e)} WHEN RUNNING COMMAND {command}!")


class TaskStatus(Enum):
    WAITING_FOR_DEPENDENCIES = 1
    DEPENDENCIES_MET = 2
    RUNNING = 3
    FINISHED_SUCCESS = 4
    FINISHED_ERROR = 5
    CANNOT_START = 6
    NOT_CHANGED = 7


class Task:
    def __init__(
        self,
        task_id: int,
        pipeline_name: str,
        instance_name: str,
        stage_name: str,
        parameterized_command: str,
        instance_parameters: Dict = None,
        track_changes: Dict = None,
        depends: List["Task"] = None,
        cpus: int = 0,
        gpus: int = 0,
        files_to_clear: List[str] = None,
    ):

        self.task_id = task_id
        self.pipeline_name = pipeline_name
        self.instance_name = instance_name
        self.stage_name = stage_name
        self.depends = depends if depends is not None else []
        self.cpus = cpus
        self.gpus = gpus
        self.files_to_clear = files_to_clear if files_to_clear is not None else []
        self.parameterized_command = parameterized_command
        self.instance_parameters = {} if instance_parameters is None else instance_parameters
        self.track_changes = {} if track_changes is None else track_changes

        self.command = substitute_parameters(self.parameterized_command, self.instance_parameters)

        self.dependencies_hashes_file = (
            f"{args.pipeline}_data_/hashes/{self.pipeline_name}__{self.instance_name}__{self.stage_name}.hashes"
        )

        self.assigned_gpus = []

        self.name = f"{self.instance_name}/{self.stage_name}"

        self.status = None
        self.update_status()

        logging.info(
            f"[Task {self.name}] Created with status {self.status} and dependencies {[d.name for d in self.depends]}."
        )

        self.exec_process = None
        self.exec_returncode = None

    def __str__(self):
        if self.status == TaskStatus.WAITING_FOR_DEPENDENCIES:
            status = colored("Depends not met ".ljust(17, "."), "yellow")
        elif self.status == TaskStatus.DEPENDENCIES_MET:
            status = colored("Ready to start ".ljust(17, "."), "yellow")
        elif self.status == TaskStatus.RUNNING:
            if len(self.assigned_gpus) > 0:
                gpus = f"on GPU {','.join([str(g) for g in self.assigned_gpus])}"
            else:
                gpus = ""
            status = colored(f"Running {gpus}".ljust(17, "."), "yellow", attrs=["blink"])
        elif self.status == TaskStatus.FINISHED_SUCCESS:
            status = colored("Finished ".ljust(17, "."), "green")
        elif self.status == TaskStatus.FINISHED_ERROR:
            status = colored("Error ".ljust(17, "."), "red")
        elif self.status == TaskStatus.CANNOT_START:
            status = colored("Cannot start ".ljust(17, "."), "red")
        elif self.status == TaskStatus.NOT_CHANGED:
            status = colored("Hasn't changed ".ljust(17, "."), "blue")
        else:
            status = colored("UNKNOWN STATUS", "red")
        return f"{status} {self.instance_name}/{self.stage_name} "

    def update_status(self):
        new_status = self.status
        if self.status not in [
            TaskStatus.RUNNING,
            TaskStatus.FINISHED_SUCCESS,
            TaskStatus.FINISHED_ERROR,
            TaskStatus.CANNOT_START,
            TaskStatus.NOT_CHANGED,
        ]:
            if all([d.status in [TaskStatus.FINISHED_SUCCESS, TaskStatus.NOT_CHANGED] for d in self.depends]):
                logging.info(f"[Task {self.name}] All dependencies met: {[d.name for d in self.depends]}.")
                if self.no_need_to_rerun():
                    new_status = TaskStatus.NOT_CHANGED
                else:
                    new_status = TaskStatus.DEPENDENCIES_MET

            elif any([d.status == TaskStatus.FINISHED_ERROR for d in self.depends]):
                new_status = TaskStatus.CANNOT_START
            else:
                new_status = TaskStatus.WAITING_FOR_DEPENDENCIES
                logging.info(
                    f"[Task {self.name}] Waiting for: {[d.name for d in self.depends if d.status not in [TaskStatus.FINISHED_SUCCESS, TaskStatus.NOT_CHANGED]]}."
                )

        if new_status != self.status:
            logging.info(f"[Task {self.name}] Changed status from {self.status} to {new_status}.")
            self.status = new_status

    def clear_files(self):
        # clear the files that need clearing
        for path in self.files_to_clear:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                logging.info(f"[Task {self.name}] Cleared file {path}.")
            elif os.path.isdir(path):
                shutil.rmtree(path)
                logging.info(f"[Task {self.name}] Cleared directory {path}.")

    def launch_task(self, logs_dir, cuda_devices=""):
        assert self.status == TaskStatus.DEPENDENCIES_MET, "Can only start a task if it's in DEPENDENCIES_MET state!"

        log_name = os.path.join(
            logs_dir,
            f"{self.pipeline_name}__{self.instance_name}__{self.stage_name}.log",
        )

        # first clear the files that need clearing
        self.clear_files()

        self.exec_returncode = multiprocessing.Value("i", -1)
        self.exec_process = multiprocessing.Process(
            target=worker,
            args=(self.command, self.exec_returncode, log_name, cuda_devices),
        )
        self.exec_process.start()
        self.status = TaskStatus.RUNNING

        logging.info(f"[Task {self.name}] Task started.")

        return self.exec_process

    def close(self):
        if self.status == TaskStatus.RUNNING:
            assert not self.exec_process.is_alive(), "Cannot close alive task!"
            self.exec_process.close()
            logging.info(f"[Task {self.name}] Task closed with return code {self.exec_returncode.value}.")
            if self.exec_returncode.value == 0:
                self.status = TaskStatus.FINISHED_SUCCESS

                # save the hashes of the dependencies so that we know not to rerun the Task
                # next time unless a dependency or parameter has changed
                os.makedirs(os.path.dirname(self.dependencies_hashes_file), exist_ok=True)
                with open(self.dependencies_hashes_file, "w") as f:
                    yaml.dump(
                        get_dependecies_hash(self.instance_parameters, self.track_changes),
                        f,
                    )

                logging.info(f"[Task {self.name}] Hashes saved to {self.dependencies_hashes_file}.")
            else:
                self.status = TaskStatus.FINISHED_ERROR
        else:
            print(f"Attempted to close not running task {str(self)}.")

        logging.info(f"[Task {self.name}] Status after closing {self.status}.")

    def no_need_to_rerun(self):
        # if the hashed dependencies and parameters are the same as the current state
        # and none of the predecessor tasks will be ran, then skip

        logging.info(f"[Task {self.name}] Checking if OK to skip.")
        dependencies_hashes = get_dependecies_hash(self.instance_parameters, self.track_changes)

        if os.path.exists(self.dependencies_hashes_file):
            logging.info(f"[Task {self.name}] Dependencies hashes file exists.")
            with open(self.dependencies_hashes_file, "r") as f:
                current_hashes = yaml.safe_load(f)
            if isinstance(current_hashes, dict):
                if dependencies_hashes == current_hashes:
                    logging.info(
                        f"[Task {self.name}] Dependencies are the same: {dependencies_hashes} and {current_hashes}."
                    )
                    logging.info(f"[Task {self.name}] Determining OK to skip")
                    return True
                else:
                    logging.info(
                        f"[Task {self.name}] Determining NOT OK to skip because hashes differ: {difference_between_dicts(current_hashes, dependencies_hashes)}."
                    )
                    return False
            else:
                logging.info(f"[Task {self.name}] Determining NOT OK to skip because the hashes file is not valid.")
                return False
        else:
            logging.info(f"[Task {self.name}] Determining NOT OK to skip because hashes file doesn't exist yet.")
            return False


class TaskList(list):
    def find_by_instance_stage(self, instance_name: str, stage_name: str):
        for t in self:
            if t.instance_name == instance_name and t.stage_name == stage_name:
                return t
        else:
            raise RuntimeError(f"Task {instance_name} -- {stage_name} not found in TaskList!")

    def append(self, item):
        assert item.task_id not in [t.task_id for t in self], "Cannot have identical task_id values!"
        assert all([dependency in self for dependency in item.depends]), "Not all dependencies are alrady added!"
        super().append(item)


class Scheduler:
    def __init__(self, tasks: List[Task], cpus: int, gpus: List[int], logs_dir: str):
        self.tasks = tasks
        self.finished = False
        self.available_cpus = cpus
        self.available_gpus = gpus
        self.logs_dir = logs_dir

        self.free_cpus = cpus
        self.free_gpus = gpus

        self.n_done = 0

        # self.tasks_order = [i for _, _, i in sorted([(t.gpus, t.cpus, i) for i, t in enumerate(self.tasks)]) ]
        self.tasks_order = list(range(len(self.tasks)))

        logging.info(f"[Scheduler] Initialized.")
        logging.info(f"[Scheduler] Task order.")
        for j, i in enumerate(self.tasks_order):
            logging.info(f"[Scheduler]  {j:>3}. {self.tasks[i].name}.")

    def step(self):
        if self.finished:
            return

        # close finished processes and free up their resources
        for task in self.tasks:
            if task.status == TaskStatus.RUNNING and not task.exec_process.is_alive():
                task.close()
                self.free_cpus += task.cpus
                self.free_gpus += task.assigned_gpus

                logging.info(f"[Scheduler] Closed {task.name}.")
                logging.info(f"[Scheduler] Free resources: cpus: {self.free_cpus}, gpus: {self.free_gpus}.")

        # launch tasks if possible (in order of most gpu and cpu requirements first):
        for i_task in self.tasks_order:
            task = self.tasks[i_task]

            task.update_status()
            if task.status == TaskStatus.DEPENDENCIES_MET:
                if self.free_cpus >= task.cpus and len(self.free_gpus) >= task.gpus:
                    self.free_cpus -= task.cpus
                    task.assigned_gpus = self.free_gpus[: task.gpus]
                    self.free_gpus = self.free_gpus[task.gpus :]
                    task.launch_task(
                        logs_dir=self.logs_dir,
                        cuda_devices=",".join([str(d) for d in task.assigned_gpus]),
                    )

                    logging.info(f"[Scheduler] Launched {task.name} on gpus {task.assigned_gpus}.")
                    logging.info(f"[Scheduler] Free resources: cpus: {self.free_cpus}, gpus: {self.free_gpus}.")
                else:
                    logging.info(f"Resouces not sufficient for {task.name}: {self.free_cpus} cpus of {task.cpus}, {len(self.free_gpus)} gpus of {task.gpus}")

        # if no tasks left to run set finished to true
        self.n_done = sum(
            [
                t.status
                in [
                    TaskStatus.FINISHED_SUCCESS,
                    TaskStatus.FINISHED_ERROR,
                    TaskStatus.CANNOT_START,
                    TaskStatus.NOT_CHANGED,
                ]
                for t in self.tasks
            ]
        )

        if self.n_done == len(self.tasks):
            self.finished = True
            logging.info(f"[Scheduler] FINISHED.")


def substitute_parameters(item, instance_parameters):
    item = copy.deepcopy(item)
    if isinstance(item, dict):
        for k, v in item.items():
            item[k] = substitute_parameters(v, instance_parameters)
    elif isinstance(item, list):
        for i in range(len(item)):
            item[i] = substitute_parameters(item[i], instance_parameters)
    elif isinstance(item, str):
        for p_name, p_value in instance_parameters.items():
            item = item.replace(f"<{p_name}>", str(p_value))
    else:
        raise RuntimeError(f"Unsupported type {type(item)}!")

    return item


def get_dependecies_hash(instance_parameters, track_changes):

    # first do parameter substitutions in the track_changes if necessary:
    track_changes = substitute_parameters(track_changes, instance_parameters)

    hashes = {}
    hashes["parameter_values"] = {k: sha256(str(v).encode()).hexdigest() for k, v in instance_parameters.items()}
    hashes["track_changes"] = {}
    for object_to_track in track_changes:
        if os.path.exists(object_to_track):
            if os.path.isfile(object_to_track):
                files_to_hash = [object_to_track]
            elif os.path.isdir(object_to_track):
                files_to_hash = []
                for path, subdirs, files in os.walk(object_to_track):
                    for name in files:
                        files_to_hash.append(os.path.join(path, name))

            files_to_hash.sort()

            sha256hash = sha256()
            for file in files_to_hash:
                with open(file, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256hash.update(chunk)

            hashes["track_changes"][object_to_track] = sha256hash.hexdigest()
        else:
            hashes["track_changes"][object_to_track] = "Doesn't exist"

    return hashes


def _flatten_dict_gen(d, parent_key, sep):
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            yield from flatten_dict(v, new_key, sep=sep).items()
        else:
            yield new_key, v


def flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = "."):
    return dict(_flatten_dict_gen(d, parent_key, sep))


def difference_between_dicts(first_dict, second_dict):

    first_dict = flatten_dict(copy.deepcopy(first_dict))
    second_dict = flatten_dict(copy.deepcopy(second_dict))

    r = ""
    diff = set(first_dict.keys()) - set(second_dict.keys())
    if len(diff) > 0:
        r += f"removed keys: {list(diff)} "

    diff = set(second_dict.keys()) - set(first_dict.keys())
    if len(diff) > 0:
        r += f"added keys: {list(diff)} "

    for k in set(first_dict.keys()).intersection(set(second_dict.keys())):
        if first_dict[k] != second_dict[k]:
            r += f"{k} changed from {first_dict[k]} to {second_dict[k]} "

    return r


# check if the pipeline files exist
if not os.path.exists(args.pipeline + ".pipeline") or not os.path.exists(args.pipeline + ".instances"):
    raise RuntimeError(f"Missing pipeline definition or instances definition file for {args.pipeline}!")

# load the configs
with open(args.pipeline + ".pipeline", "r") as f:
    pipeline_description = yaml.safe_load(f)
    logging.info(f"[pipeline_description] {pipeline_description}.")
with open(args.pipeline + ".instances", "r") as f:
    instances_description = yaml.safe_load(f)
    logging.info(f"[instances_description] {instances_description}.")

console = Console()

tasks = TaskList()
task_idx = 0
for instance_name, instance_parameters in instances_description.items():
    for stage_name, stage in pipeline_description["stages"].items():

        # if we want to run only one task and this is not it, skip adding it
        if args.single_tasks is not None:
            patterns = args.single_tasks.split(",")
            if not any([fnmatch.fnmatch(f"{instance_name}/{stage_name}", pattern) for pattern in patterns]):
                continue

        # if args.single_tasks is not None and f"{instance_name}/{stage_name}" not in args.single_tasks.split(","):
        #     continue

        if args.single_tasks is not None:
            dependencies = []
        else:
            dependencies = [
                tasks.find_by_instance_stage(instance_name=instance_name, stage_name=d_name)
                for d_name in stage.get("wait", [])
            ]

        clear_requests = substitute_parameters(stage.get("clear", []), instance_parameters)

        tasks.append(
            Task(
                task_id=task_idx,
                pipeline_name=args.pipeline.split("/")[-1],
                instance_name=instance_name,
                stage_name=stage_name,
                parameterized_command=stage["command"],
                instance_parameters=instance_parameters,
                depends=dependencies,
                track_changes=stage["track_changes"],
                cpus=stage.get("cpus", 0),
                gpus=stage.get("gpus", 0),
                files_to_clear=clear_requests,
            )
        )
        task_idx += 1

        # Update statuses
        console.clear()
        console.print(Text(colored(f"Status for {args.pipeline} pipeline:\n", attrs=["bold"])))
        statuses = [Text.from_ansi(str(t)) for t in tasks]
        console.print(Columns(statuses))

if args.single_tasks is not None and len(tasks) == 0:
    raise RuntimeError(f"When single task is selected we expect to have at least one task to run!")

scheduler = Scheduler(tasks=tasks, cpus=args.cpus, gpus=args.gpus.split(","), logs_dir=logs_dir)

if args.just_show:
    console.clear()
    console.print(Text(colored("Commands to be run:", attrs=["bold"])))
    for t in tasks:
        console.print(f"{t.name}:\n{t.command}\n")
    exit()

# save initial situation
with open(f"{logs_dir}/start", "w") as f:
    for t in tasks:
        f.write(ansi_escape.sub("", str(t)) + "\n")

while not scheduler.finished and not args.dry_run:

    scheduler.step()

    console.clear()
    console.print(
        Text(
            colored(
                f"Status for {args.pipeline} pipeline: (Done {scheduler.n_done}/{len(scheduler.tasks)})\n",
                attrs=["bold"],
            )
        )
    )
    statuses = [Text.from_ansi(str(t)) for t in tasks]
    console.print(Columns(statuses))
    time.sleep(1)

with open(f"{logs_dir}/end", "w") as f:
    for t in tasks:
        f.write(ansi_escape.sub("", str(t)) + "\n")
