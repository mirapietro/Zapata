"""Zeus utility functions

This Python module provides some basic utility functions to interact with
the Zeus SuperComputer at CMCC. It manages file upload and download,
submission of scripts on LSF and execution of general bash commands.

The module has been tested with Python 3.6, but it should work with most
Python 3 versions. It requires IPython and ipywidgets modules; to install
the dependencies run for example:

    pip3 install IPython ipywidgets

Import the module using:

    import zeus_util as zeus

Before calling any other function from this module run the `init` function.

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

# Global variable to store the login username
username = ""
# Global variable with the IP address of the Zeus login node
hostname = "192.168.118.11"


def init(user):
    """Initialize the module with the Zeus login username.

    This function must be called before any other function from the module.

    Parameters
    ----------
    user : str
         Username on Zeus.

    Returns
    -------
    None

    Examples
    --------
    >>> zeus.init('username')

    """
    global username
    username = user
    return None


def _remote_cmd(command, out=None):
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


def _remote_scp(src, dst, way, out=None):
    if way == "get":
        cmd = [
            "scp",
            "-r",
            "{user}@{host}:{src}".format(
                user=username, host=hostname, src=src
            ),
            "{dst}".format(dst=dst),
        ]
    elif way == "put":
        cmd = [
            "scp",
            "-r",
            "{src}".format(src=src),
            "{user}@{host}:{dst}".format(
                user=username, host=hostname, dst=dst
            ),
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


def _remote_bsub(script_path, out):
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
        out.append_stdout(stdout_line.strip() + "\n")
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        for stdout_line in iter(popen.stderr.readline, ""):
            out.append_stdout(stdout_line.strip() + "\n")
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
            if _remote_cmd("mkdir -p ~/.tmp", out):
                out.append_stdout(
                    "Something went wrong while compressing the file"
                )
                return None
            if _remote_cmd(
                "tar -zcf ~/.tmp/"
                + os.path.basename(data)
                + ".tar.gz "
                + data,
                out,
            ):
                out.append_stdout(
                    "Something went wrong while compressing the file"
                )
                return None

            local_path = os.path.join(
                os.environ["PWD"], os.path.basename(data) + ".tar.gz"
            )
            remote_path = "~/.tmp/" + os.path.basename(data) + ".tar.gz "
        else:
            local_path = os.path.join(
                os.environ["PWD"], os.path.basename(data)
            )
            remote_path = data

        if _remote_scp(remote_path, local_path, "get", out):
            out.append_stdout(
                "Something went wrong while getting the output data from Zeus"
            )
            return None

        if compress:
            _remote_cmd("rm ~/.tmp/" + os.path.basename(data) + ".tar.gz", out)

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
