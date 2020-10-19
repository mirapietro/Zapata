"""

This module provides core utility functions to interact with the Zeus cluster at CMCC Supercomputing Center. 

It manages file upload and download, submission of scripts on LSF, execution of general bash commands and creation and shutdown of Dask clusters.

**USAGE NOTE**: Before calling any other function from this module run the `init` function.

"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import subprocess
import os
import re
import threading
import time
from IPython.display import display
import ipywidgets as widgets
import math
from dask.distributed import Client

# Global variable to store the login username
username = ""
# Global variable with the hostname to login
hostname = ""
# Global variable to store the user home path
home = ""
# Global variable to store the user tmp folder
tmp_path = ""
# Global variable to store the conda environment to use
conda_env = "data-science-cmcc-v1"


def init(user, host=None, remote_env=None):
    """Initialize the module with the Zeus login username.

    It inizializes the module and checks if the ssh connection can be made.
    This function must be called before any other function from the module.

    Parameters
    ----------
    user : str
         Username on Zeus.
    host : str
         Local hostname for Zeus cluster (default: 192.168.118.11).
    remote_env : str
         Name of the conda environment to use on Zeus (default: data-science-cmcc-v1).

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.init('username')

    """
    global username
    global hostname
    global home
    global tmp_path
    username = user
    if host is None:
        hostname = "192.168.118.11"
    else:
        hostname = host

    if remote_env:
        set_env(remote_env)

    ret, res = _remote_cmd("echo $HOME", None, True)
    if ret == 0:
        home = res
        tmp_path = "%s/.tmp/" % home
    else:
        print(
            "Error in connection to Zeus cluster. Check provided username, "
            "network/VPN connection and ssh key setup"
        )
    return None


def set_env(env_name):
    """Set the name of the conda environment to use remotely on Zeus.

    Parameters
    ----------
    env_name : str
         Name of the conda environment to use.

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.set_env('condaenv')

    """
    global conda_env
    ret = _remote_cmd(
        "module load anaconda/3.7; conda env list | grep %s" % env_name,
        None,
        False,
    )
    if ret == 0:
        conda_env = env_name
    else:
        print("Unable to find specified conda env. Using default one.")
    return None


def _remote_cmd(command, out=None, var=False):
    cmd = [
        "ssh",
        "{user}@{host}".format(user=username, host=hostname),
        "{cmd}".format(cmd=command),
    ]
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        if out:
            out.append_stdout(stdout_line.strip() + "\n")
        elif var is True:
            res = stdout_line.strip()
        else:
            print(stdout_line.strip())
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        if out:
            for stdout_line in iter(popen.stderr.readline, ""):
                out.append_stdout(stdout_line.strip() + "\n")
        else:
            for stdout_line in iter(popen.stderr.readline, ""):
                print(stdout_line.strip())

    if var is True:
        return return_code, res
    else:
        return return_code


def _remote_scp(src, dst, way, out=None):
    if way == "get":
        cmd = [
            "scp",
            "-r",
            "{user}@{host}:{src}".format(user=username, host=hostname, src=src),
            "{dst}".format(dst=dst),
        ]
    elif way == "put":
        cmd = [
            "scp",
            "-r",
            "{src}".format(src=src),
            "{user}@{host}:{dst}".format(user=username, host=hostname, dst=dst),
        ]
    else:
        if out:
            out.append_stdout("Scp wrapper not used properly\n")
        else:
            print("Scp wrapper not used properly\n")
        return -1

    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        if out:
            out.append_stdout(stdout_line.strip() + "\n")
        else:
            print(stdout_line.strip())
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        if out:
            for stdout_line in iter(popen.stderr.readline, ""):
                out.append_stdout(stdout_line.strip() + "\n")
        else:
            for stdout_line in iter(popen.stderr.readline, ""):
                print(stdout_line.strip())
    return return_code


def _remote_bsub(script_path, out=None):
    cmd = [
        "ssh",
        "{user}@{host}".format(user=username, host=hostname),
        "bsub < {cmd}".format(cmd=script_path),
    ]
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    jobid = -1
    for stdout_line in iter(popen.stdout.readline, ""):
        if "is submitted to queue" in stdout_line:
            res = re.match(r"^.*<([0-9]*)>.*$", stdout_line)
            if res:
                jobid = res.group(1)
        if out:
            out.append_stdout(stdout_line.strip() + "\n")
        else:
            print(stdout_line.strip() + "\n")
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        for stdout_line in iter(popen.stderr.readline, ""):
            if out:
                out.append_stdout(stdout_line.strip() + "\n")
            else:
                print(stdout_line.strip() + "\n")
    return return_code, jobid


def _remote_bjobs(jobid, out):
    status_str = "bjobs -o stat {jobid} | tail -n 1".format(jobid=jobid)
    cmd = [
        "ssh",
        "{user}@{host}".format(user=username, host=hostname),
        "{cmd}".format(cmd=status_str),
    ]
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    status = None
    for stdout_line in iter(popen.stdout.readline, ""):
        status = stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        for stdout_line in iter(popen.stderr.readline, ""):
            out.append_stdout(stdout_line.strip() + "\n")
    return return_code, status


def execute(cmd):
    """Execute a command remotely on Zeus.

    Parameters
    ----------
    cmd : str
         Command to be executed on Zeus.

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.execute('ls ~/')

    """
    _remote_cmd(cmd)
    return None


def get(src, dst):
    """Download a file from Zeus.

    Copy remote file specified in `src` from Zeus to the local path
    specified in `dst`.

    Parameters
    ----------
    src : str
         Path of the remote file to be downloaded from Zeus.
    dst : str
         Local path/name where to copy the file from Zeus.

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.get('/users_home/user/file.nc', '/home/user/file.nc')

    """
    _remote_scp(src, dst, "get")
    return None


def put(src, dst):
    """Upload a file to Zeus.

    Copy a local file specified in `src` to the remote path specified in
    `dst` on Zeus.

    Parameters
    ----------
    src : str
         Path of the local file to be uploaded on Zeus.
    dst : str
         Path/name where to copy the file on Zeus.

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.put('/home/user/file.nc', '/users_home/user/file.nc')

    """
    _remote_scp(src, dst, "put")
    return None


def process(script_name, data, compress=False, frequency=10):
    """Submit a script on LSF and download the output data.

    This function uploads the local script specified in `script_name`,
    executes the script on LSF (i.e., bsub < script_name) and downloads the
    output specified in `data` once the job is completed.

    Output data can be optionally compressed as tar.gz before downloading it.
    Job status checking frequency can be adjusted based on the expected job
    duration.

    The function is non-blocking, which means that other cells in the notebook
    can be executed right after this function is started, without needing to
    wait for the job to end.

    Parameters
    ----------
    script_name : src
        Absolute (local) path of the script to be submitted on LSF.
    data : src
        Absolute (remote) path of data to be downloaded.
    compress : bool, optional
        If data downloaded has to be compressed or not (default is False).
    frequency : int, optional
        Interval for checking job status in seconds (default is 10s).

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.process('/home/user/script.lsf', '/users_home/user/output.nc',
    compress=True, frequency=20)

    """
    # Perform initial checks
    if not os.path.isabs(script_name) or not os.path.isfile(script_name):
        print("Please specify the absolute path of the script file")
        return None

    if not os.path.isabs(data):
        print("Please specify the absolute path of the remote data file")
        return None

    def thread_func(script_name, data, compress, out):
        if _remote_scp(script_name, os.path.basename(script_name), "put", out):
            out.append_stdout(
                "Something went wrong while uploading the script on Zeus"
            )
            return None

        ret, jobid = _remote_bsub(os.path.basename(script_name), out)
        if ret or int(jobid) < 0:
            out.append_stdout(
                "Something went wrong while running the script on LSF"
            )
            return None

        old_status = ""
        while True:
            ret, status = _remote_bjobs(jobid, out)
            if ret or not status:
                out.append_stdout(
                    "Something went wrong while checking the job status on LSF"
                )
                return None
            if status and status != old_status:
                out.append_stdout("Job status is: " + status)
                old_status = status
                if "DONE" in status or "EXIT" in status:
                    break

            time.sleep(frequency)

        if "EXIT" in status:
            out.append_stdout("Job execution was unsuccessful")
            return None

        if _remote_cmd("ls " + data, out):
            out.append_stdout("Remote data path not found")
            return None

        if compress:
            if _remote_cmd("mkdir -p %s" % tmp_path, out):
                out.append_stdout(
                    "Something went wrong while compressing the file"
                )
                return None
            if _remote_cmd(
                "tar -zcf %s%s.tar.gz %s"
                % (tmp_path, os.path.basename(data), data),
                out,
            ):
                out.append_stdout(
                    "Something went wrong while compressing the file"
                )
                return None

            local_path = os.path.join(
                os.environ["PWD"], os.path.basename(data) + ".tar.gz"
            )
            remote_path = os.path.join(
                tmp_path, os.path.basename(data) + ".tar.gz "
            )
        else:
            local_path = os.path.join(os.environ["PWD"], os.path.basename(data))
            remote_path = data

        if _remote_scp(remote_path, local_path, "get", out):
            out.append_stdout(
                "Something went wrong while getting the output data from Zeus"
            )
            return None

        if compress:
            _remote_cmd(
                "rm %s%s.tar.gz" % (tmp_path, os.path.basename(data)), out
            )

        out.append_stdout("Data has been downloaded in: " + local_path)
        return None

    display("Starting Job execution")
    out = widgets.Output()
    display(out)

    thread = threading.Thread(
        target=thread_func, args=(script_name, data, compress, out)
    )
    thread.start()
    return None


def info(jobid=None):
    """Get status of LSF jobs on Zeus.

    This function returns the list of the jobs the user submitted on LSF. If
    `jobid` argument is provided then only the status of that particular job
    is shown, otherwise all recent jobs will be listed.

    Parameters
    ----------
    jobid : int, optional
         ID of LSF job.

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.info()

    >>> zeus.info(1234)

    """
    if jobid is not None:
        cmd = "bjobs " + str(jobid)
    else:
        cmd = "bjobs -a"

    _remote_cmd(cmd)
    return None


def start_dask(
    project,
    cores,
    memory,
    name="Dask-Test",
    processes=None,
    queue="p_short",
    local_directory="~/dask-space",
    interface="ib0",
    walltime=None,
    job_extra=None,
    env_extra=None,
    log_directory="~/dask-space",
    death_timeout=60,
    n_workers=1,
):
    """Start a new Dask cluster on Zeus

    This function starts a new Dask scheduler on the cluster front-end node
    and a set of worker process on the cluster compute nodes. The function
    returns a ready-to-use Dask client object. The arguments defined in the
    interface are lent from the Dask Jobqueue interface.

    Parameters
    ----------
    project : str
        Accounting string associated with each worker job. Passed to
        `#BSUB -P` option.
    cores : int
        Number of cores for the worker nodes. Passed to `#BSUB -n` option.
    memory: str
        Total amount of memory per worker job. Passed to `#BSUB -M` option.
    name: str, optional
        Name of Dask workers. By default set to Dask-test.
    processes: int, optional
        Cut the job up into this many processes. Good for GIL workloads or for
        nodes with many cores. By default, process ~= sqrt(cores) so that the
        number of processes and the number of threads per process is roughly
        the same.
    queue: str, optional
        Destination queue for each worker job. Passed to #BSUB -q option. By
        default `p_short` queue is used.
    local_directory: str, optional
        Dask worker local directory for file spilling. By default the folder
        `dask-space` in the home directory is used.
    interface: str, optional
        Network interface like `eth0` or `ib0`. This will be used for the Dask
        workers interface. By default `ib0` is used.
    walltime: str, optional
        Walltime for each worker job in HH:MM. Passed to `#BSUB -W` option. If
        not specified the default queue walltime is used.
    job_extra: list, optional
        List of optional LSF options, for example -x. Each option will be
        prepended with the #BSUB prefix.
    env_extra: list, optional
        Optional commands to add to script before launching worker.
    log_directory: str, optional
        Directory to use for job scheduler logs. By default the folder
        `dask-space` in the home directory is used.
    death_timeout: float, optional
        Seconds to wait for a scheduler before closing workers (default is 60).
    n_workers : int, optional
        Number of worker process to startup, i.e. jobs on LSF (default is 1).

    Returns
    -------
    dask.distributed.Client
        A ready-to-use Dask distributed client connected to the scheduler

    Examples
    --------
    >>> client = zeus.start_dask(
              project="R000",
              cores=36,
              memory="80 GB",
              name="Test",
              processes=12,
              local_directory="~/dask-space",
              interface="ib0",
              walltime="00:30",
              job_extra=["-x"],
              n_workers=1
             )
    Create a new cluster with a single worker on a whole Zeus node, using 12
    processes (3 threads/process), 80GB of RAM memory requested. Note that each
    process will get a maximum of 80/12GB of memory

    """

    # default values
    shebang = "#!/bin/bash"
    python = "/zeus/opt/anaconda/3.7/envs/data-science-cmcc-v1/bin/python"

    def lsf_format_bytes_ceil(n, lsf_units="mb"):
        """ Format bytes as text
        Convert bytes to megabytes which LSF requires.
        Parameters
        ----------
        n: int
            Bytes
        lsf_units: str
            Units for the memory in 2 character shorthand, kb through eb
        Examples
        --------
        >>> lsf_format_bytes_ceil(1234567890)
        '1235'
        """
        # Adapted from dask_jobqueue lsf.py
        units = {
            "B": 1,
            "KB": 10 ** 3,
            "MB": 10 ** 6,
            "GB": 10 ** 9,
            "TB": 10 ** 12,
        }
        number, unit = [string.strip() for string in n.split()]
        lsf_units = lsf_units.lower()[0]
        converter = {"k": 1, "m": 2, "g": 3, "t": 4, "p": 5, "e": 6, "z": 7}
        return "%d" % math.ceil(
            float(number) * units[unit] / (1000 ** converter[lsf_units])
        )

    def create_scheduler_script(
        shebang, python, name, log_directory, env_extra
    ):
        sched_script_lines = []
        sched_script_lines.append("%s" % shebang)
        """
        sched_script_lines.append("")
        sched_script_lines.append("#BSUB -J scheduler_%s" % name)
        sched_script_lines.append("#BSUB -e %s/scheduler_%s-%%J.err" % (log_directory, name))
        sched_script_lines.append("#BSUB -o %s/scheduler_%s-%%J.out" % (log_directory, name))
        sched_script_lines.append("#BSUB -q %s" % scheduler_queue)
        sched_script_lines.append("#BSUB -P %s" % project)

        memory_string = lsf_format_bytes_ceil(scheduler_memory)
        sched_script_lines.append("#BSUB -M %s" % memory_string)

        if scheduler_cores > 36:
            scheduler_cores = 36
            print("Worker cores specification for LSF higher than available, initializing it to %s" % scheduler_cores)
        sched_script_lines.append("#BSUB -n %s" % scheduler_cores)
        if scheduler_cores > 1:
            sched_script_lines.append('#BSUB -R "span[hosts=1]"')

        if walltime is not None:
            sched_script_lines.append("#BSUB -W %s" % walltime)

        if job_extra is not None:
            sched_script_lines.extend(["#BSUB %s" % arg for arg in job_extra])
        """
        # Zeus specific lines
        sched_script_lines.append("")
        sched_script_lines.append("module load anaconda/3.7")
        sched_script_lines.append("source activate %s" % conda_env)

        if env_extra is not None:
            sched_script_lines.extend(["%s" % arg for arg in env_extra])

        # Executable lines
        sched_exec = "%s -m distributed.cli.dask_scheduler" % python
        sched_exec += (
            " --port 0 --dashboard-address 0 --scheduler-file %s/connection"
            " --idle-timeout 3600 --local-directory %s"
            % (local_directory, local_directory,)
        )
        sched_exec += " --interface ens2f1"
        sched_exec += " >> %s/scheduler_%s.log 2>&1 &" % (log_directory, name)
        sched_script_lines.append(sched_exec)
        sched_script = "\n".join(sched_script_lines)

        return sched_script

    def create_worker_script(
        shebang,
        name,
        log_directory,
        project,
        worker_queue,
        worker_memory,
        worker_cores,
        walltime,
        job_extra,
        env_extra,
        interface,
        processes,
        death_timeout,
        sched_ip,
    ):

        if log_directory[0:1] == "~":
            log_directory = log_directory.replace("~", home)

        worker_script_lines = []
        worker_script_lines.append("%s" % shebang)
        worker_script_lines.append("")
        worker_script_lines.append("#BSUB -J dask_worker_%s" % name)
        worker_script_lines.append(
            "#BSUB -e %s/worker_%s-%%J.err" % (log_directory, name)
        )
        worker_script_lines.append(
            "#BSUB -o %s/worker_%s-%%J.out" % (log_directory, name)
        )
        worker_script_lines.append("#BSUB -q %s" % worker_queue)
        worker_script_lines.append("#BSUB -P %s" % project)

        memory_string = lsf_format_bytes_ceil(worker_memory)
        worker_script_lines.append("#BSUB -M %s" % memory_string)

        if worker_cores > 36:
            worker_cores = 36
            print(
                "Worker cores specification for LSF higher than available, "
                "initializing it to %s" % worker_cores
            )
        worker_script_lines.append("#BSUB -n %s" % worker_cores)
        if worker_cores > 1:
            worker_script_lines.append('#BSUB -R "span[hosts=1]"')

        if walltime is not None:
            worker_script_lines.append("#BSUB -W %s" % walltime)

        if job_extra is not None:
            worker_script_lines.extend(["#BSUB %s" % arg for arg in job_extra])

        # Python env specific lines
        worker_script_lines.append("")
        worker_script_lines.append("module load anaconda/3.7")
        worker_script_lines.append("source activate %s" % conda_env)

        if env_extra is not None:
            worker_script_lines.extend(["%s" % arg for arg in env_extra])

        # Executable lines
        worker_exec = "%s -m distributed.cli.dask_worker %s" % (
            python,
            sched_ip,
        )
        worker_exec += " --local-directory %s" % local_directory
        worker_exec += " --interface %s" % interface

        # Detect memory, processes and threads per each worker
        if processes is None:
            processes = max(math.floor(math.sqrt(worker_cores)), 1)
        threads = max(math.floor(float(worker_cores) / processes), 1)
        mem = float(memory_string) / processes

        worker_exec += (
            " --nthreads %i --nprocs %i --memory-limit %.2fMB --name 0 --nanny"
            " --death-timeout %i" % (threads, processes, mem, death_timeout)
        )

        worker_script_lines.append(worker_exec)
        worker_script = "\n".join(worker_script_lines)

        return worker_script

    def delete_tmp_files(local_path, remote_path):
        local_file = os.path.join(local_path, "scheduler.sh")
        if os.path.exists(local_file):
            os.remove(local_file)
        local_file = os.path.join(local_path, "worker.lsf")
        if os.path.exists(local_file):
            os.remove(local_file)
        local_file = os.path.join(local_path, "connection")
        if os.path.exists(local_file):
            os.remove(local_file)
        _remote_cmd("rm %s{%s,%s}" % (tmp_path, "scheduler.sh", "worker.lsf"))
        return None

    def run_scheduler(local_path, remote_path, local_directory, sched_script):

        timeout = 20
        local_file = os.path.join(local_path, "scheduler.sh")
        remote_file = os.path.join(remote_path, "scheduler.sh")

        with open(local_file, "w") as sched_file:
            sched_file.write(sched_script)

        if _remote_scp(local_file, remote_file, "put"):
            print("Error while copying scripts to Zeus")
            return None

        if _remote_cmd("/bin/bash %s" % remote_file):
            print("Something went wrong while executing Dask scheduler script")
            delete_tmp_files(local_path, remote_path)
            stop_dask()
            return None

        # Check connection file availablility
        i = 0
        ret = -1
        while i < timeout:
            time.sleep(1)
            ret = _remote_cmd("ls %s/connection" % local_directory)
            if ret == 0:
                break
            i += 1

        if ret != 0:
            print("Unable to retrieve Dask scheduler address")
            delete_tmp_files(local_path, remote_path)
            stop_dask()
            return None

        local_file = os.path.join(local_path, "connection")
        remote_file = "%s/connection" % local_directory
        if _remote_scp(remote_file, local_file, "get"):
            print("Error while copying files from Zeus")
            delete_tmp_files(local_path, remote_path)
            stop_dask()
            return None

        # Read connection info
        import json

        sched_address = None
        with open(local_file) as f:
            data = json.load(f)
            if "address" in data:
                sched_address = data["address"]

        if sched_address is None:
            print(
                "Something went wrong while retreiving Dask scheduler address"
            )
            delete_tmp_files(local_path, remote_path)
            stop_dask()
            return None

        return sched_address

    def run_workers(local_path, remote_path, n_workers, worker_script):

        local_file = os.path.join(local_path, "worker.lsf")
        remote_file = os.path.join(remote_path, "worker.lsf")

        with open(local_file, "w") as worker_file:
            worker_file.write(worker_script)

        if _remote_scp(local_file, remote_file, "put"):
            print("Error while copying scripts to Zeus")
            delete_tmp_files(local_path, remote_path)
            stop_dask()
            return None

        if n_workers < 1:
            n_workers = 1

        # Run worker scripts
        job_array = []
        job_num = 0
        for i in range(0, n_workers):
            if job_num > 0:
                if _remote_cmd(
                    "sed -i 's/--name %i/--name %i/g' %s"
                    % (job_num - 1, job_num, remote_file)
                ):
                    print(
                        "Something went wrong while running the script on LSF"
                    )
                    delete_tmp_files(local_path, remote_path)
                    stop_dask()
                    return None
            job_num += 1
            ret, jobid = _remote_bsub(remote_file)
            if ret or int(jobid) < 0:
                print("Something went wrong while running the script on LSF")
                delete_tmp_files(local_path, remote_path)
                stop_dask()
                return None
            else:
                job_array.append(jobid)

        return job_array

    if (
        project is None
        or cores is None
        or memory is None
        or name is None
        or queue is None
        or local_directory is None
        or log_directory is None
        or death_timeout is None
        or n_workers is None
    ):
        print("One or more arguments are not set or are set to None")
        return None

    # Create folder for local and remote scripts
    local_path = os.path.join(os.environ["HOME"], ".tmp/")
    os.makedirs(local_path, exist_ok=True)
    ret = _remote_cmd("mkdir -p %s" % tmp_path)
    if ret:
        print("Unable to create folders for Dask execution")
        return None

    # Create remote folder for dask execution
    ret = _remote_cmd("mkdir -p %s" % local_directory)
    if ret:
        print("Unable to create folders for Dask execution")
        return None
    if local_directory is not log_directory:
        ret = _remote_cmd("mkdir -p %s" % log_directory)
        if ret:
            print("Unable to create folders for Dask logs")
            return None

    sched_script = create_scheduler_script(
        shebang, python, name, log_directory, env_extra
    )

    sched_address = run_scheduler(
        local_path, tmp_path, local_directory, sched_script
    )
    if sched_address is None:
        return None
    print("Scheduler address is: %s" % sched_address)

    worker_script = create_worker_script(
        shebang,
        name,
        log_directory,
        project,
        queue,
        memory,
        cores,
        walltime,
        job_extra,
        env_extra,
        interface,
        processes,
        death_timeout,
        sched_address,
    )

    job_array = run_workers(local_path, tmp_path, n_workers, worker_script)
    if len(job_array) == 0:
        return None

    # Remove all tmp files
    delete_tmp_files(local_path, tmp_path)

    client = Client(sched_address)
    return client


def stop_dask(client=None):
    """Stop a running Dask cluster on Zeus

    This function stops a running Dask scheduler on the cluster, both
    scheduler and worker processes. Note that this function will stop every
    Dask cluster running under the provided username on Zeus.

    Parameters
    ----------
    client : dask.distributed.Client, optional
        Optional Dask Client refering to a running Dask cluster

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.stop_dask(client)

    """

    import warnings

    warnings.filterwarnings("ignore")

    if client is not None:
        client.shutdown()
        client.close()

    _remote_cmd("pkill -f 'distributed.cli.dask_scheduler'")
    _remote_cmd("bkill -J dask_worker*")

    return None


def run_dask(client, fn_name, args, timeout=60, frequency=1):
    """Run a function on a Dask cluster on Zeus

    This function submits the execution of the function `fn_name` with
    arguments `args` on the Dask cluster associated to `client`. The result
    can be accessed with the `.result()` method.

    `client` is the result of the Dask cluster deployed on Zeus with
    `start_dask` function. If the cluster is not yet ready, this function
    will periodically check the status every `freqeuncy` seconds to check
    if `fn_name` can be submitted. If the `timeout` value in seconds is
    reached before the cluster is ready, the function will simply abort the
    execution.

    The function is non-blocking, which means that other cells in the notebook
    can be executed right after this function is started, without needing to
    wait for the function to end.

    Parameters
    ----------
    client : dask.distributed.Client
        Dask Client refering to a running Dask cluster
    fn_name : function
        Function to be executed remotely on the Dask cluster
    args : tuple
        Tuple with arguments to be passed to the function `fn_name`
    timeout : int, optional
        Timeout in seconds for checking the cluster status, after which the
        execution is aborted (default is 60s).
    frequency : int, optional
        Interval for checking Dask cluster status in seconds (default is 1s).

    Returns
    -------
    <locals>.async_submit
        A pointer to the thread executing the function

    Examples
    --------
    >>> def simple_function(a,b):
        ... c = a + b
        ... return c

    Start function execution
    >>> res = run_dask(client, simple_function, (1,2))

    Get result from the execution
    >>> print(res.result())
    """

    class async_submit(threading.Thread):
        def __init__(self, args):
            threading.Thread.__init__(self, args=args)
            self.res = None
            self.exception = None

        def run(self):
            try:
                self.res = thread_func(*self._args)
            except Exception as e:
                self.exception = e

        def result(self):
            threading.Thread.join(self)
            if self.exception:
                raise self.exception
            return self.res

    def thread_func(client, timeout, frequency, out, fn_name, args):

        from distributed import get_client

        def remote_fn(fn_name, args):
            import dask

            client = get_client()
            return fn_name(*args)

        # Check if Dask worker is available
        import time

        iterations = int(timeout / frequency)
        exec_flag = False
        result = None
        for i in range(0, iterations):
            if (
                "workers" in client._scheduler_identity
                and client._scheduler_identity["workers"]
            ):
                out.append_stdout("Execution started on remote Dask cluster\n")
                exec_flag = True
                future = client.submit(remote_fn, fn_name, args)
                try:
                    result = future.result()
                except Exception as e:
                    out.append_stdout(
                        "Error during execution on Dask. Access the result"
                        " method to get the full error stack.\n"
                    )
                    out.append_stderr("ERROR: %sn" % str(e))
                    raise
                break
            else:
                time.sleep(frequency)

        if exec_flag:
            out.append_stdout("Function execution completed\n")
        else:
            out.append_stdout(
                "Unable to run the function within the provided timeout\n"
            )

        return result

    # Check input type
    if not isinstance(client, Client):
        print("client is not a valid Dask client")
        return None

    if not callable(fn_name):
        print("fn_name is not a valid function")
        return None

    if not isinstance(args, tuple) or not args:
        print("args is not a valid tuple")
        return None

    display("Starting Remote Dask function")
    out = widgets.Output()
    display(out)

    thread = async_submit(args=(client, timeout, frequency, out, fn_name, args))
    thread.start()
    return thread
