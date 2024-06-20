import contextlib
import zipfile

from flask import Flask, request, jsonify, send_from_directory, after_this_request
from celery import Celery
from .workflow import build_fragment_library, build_gmx_parameter_lib, valid_gmx_parameter_lib, patch_gmx_top
import os
from pathlib import Path
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

allowed_suffixes = [".sdf", ".mol"]

# 定义一个目录来存储结果文件
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)


@contextlib.contextmanager
def working_directory(path):
    """
    A context manager which changes the working directory to the given
    path, and then changes it back to its previous value on exit.
    """
    prev_cwd = os.getcwd()
    os.chdir(path)
    yield
    os.chdir(prev_cwd)


@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('file')  # 'file' 是前端上传时用的字段名
    if not files:
        return jsonify({'error': 'No files provided'}), 400

    task_id = str(celery.uuid())
    task_path = Path(RESULTS_DIR) / task_id
    mol_path = task_path / "molecules"
    mol_path.mkdir(exist_ok=True, parents=True)

    mol_files = []
    for file in files:
        if file and Path(file.filename).suffix in allowed_suffixes:
            secure_name = secure_filename(file.filename)
            file_path = mol_path / secure_name
            file.save(str(file_path))
            mol_files.append(secure_name)

    if not mol_files:  # If no allowed files were saved
        return jsonify({'error': 'No allowed files were uploaded'}), 400

    result = main.apply_async((task_id, mol_files), task_id=task_id)

    return jsonify({'message': 'Files uploaded successfully', 'results': result}), 200


@celery.task
def main(task_id, mol_files, model_type="DP-GFN2-xTB", model_file="/root/model-gfn2-dpv3.pt"):
    task_path = Path(RESULTS_DIR) / task_id
    task_path.mkdir(exist_ok=True, parents=True)

    mol_path = task_path / "molecules"
    molpatch_path = task_path / "molecules_patched"
    mol_path.mkdir(exist_ok=True)
    molpatch_path.mkdir(exist_ok=True)

    with working_directory(task_path):

        print("Prepare Calculator")
        # load QM calculator
        from .calculators.dp import DPCalculator, DPTBCalculator

        if model_type == "DP-GFn2-xTB":
            qm_calc = DPTBCalculator(model_file, tb_method="GFN2-xTB")
        else:
            qm_calc = DPCalculator(model_file)

        # set rerun calculators
        rerun_calc = None
        # from ase.calculators.psi4 import Psi4
        # rerun_calc = Psi4(method="wb97x-d", basis="def2-svp", memory="2GB", num_threads=4)

        print("Build Fragment Library")
        build_fragment_library(
            [f"molecules/{mol}" for mol in mol_files],
            qm_calc,
            recalc_calculator=rerun_calc
        )

        print("Build GMX Parameter Library")
        build_gmx_parameter_lib(parameter_lib="param.pkl")

        print("Valid GMX Parameter Library")
        valid_gmx_parameter_lib(parameter_lib="param.pkl")

        print("Patch GMX Topology")
        if not os.path.exists("molecules_patched"):
            os.makedirs("molecules_patched")
        for mol_name in mol_files:
            name = mol_name.split(".")[0]
            patch_gmx_top(
                f"molecules/{mol_name}",
                "param.pkl",
                "tmp.top",
                f"molecules_patched/{name}.top"
            )
            os.remove("tmp.top")
    return "Success!"


@app.route('/status/<task_id>', methods=['GET'])
def task_status(task_id):
    task = main.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Pending...'
        }
    elif task.state != 'FAILURE':
        if task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'result': task.result,
                'file_url': f'/download/{task.result}'
            }
        else:
            response = {
                'state': task.state,
                'result': task.result
            }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)  # 异常信息
        }
    return jsonify(response)


@app.route('/download/<task_id>/', methods=['GET'])
def download_files(task_id):
    directory = os.path.join(RESULTS_DIR, task_id)
    if not os.path.exists(directory):
        return jsonify({'error': 'Task not found.'}), 404

    # Create a ZIP file
    zip_filename = f"{task_id}.zip"
    zip_path = os.path.join(RESULTS_DIR, zip_filename)

    # Remove existing zip file if it exists to avoid adding duplicate files in resending the request
    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, start=directory))

    # Serve the ZIP file
    @after_this_request
    def remove_file(response):
        try:
            os.remove(zip_path)
            # shutil.rmtree(directory)  # Optionally remove the directory after download
        except Exception as error:
            app.logger.error("Error removing or closing downloaded file handle", error)
        return response

    return send_from_directory(directory=RESULTS_DIR, path=zip_filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
