import sys
import os
import argparse

import uvicorn
import json

from dotenv import load_dotenv

load_dotenv()

from fastapi.middleware.cors import CORSMiddleware


# Add the path to the directory containing the module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastchat.utils import build_logger
from fastchat.serve import controller as controller_module
from fastchat.serve.controller import Controller, app as controller_app
from fastchat.modules.gptq import GptqConfig
from fastchat.serve import model_worker as model_worker_module
from fastchat.serve.model_worker import ModelWorker, app as worker_app, worker_id
from fastchat.model.model_adapter import add_model_args
from fastchat.serve.openai_api_server import app as api_app, app_settings

logger = build_logger("main", "main.log")

# docs/langchain_integration.md

# launch the controller
def launch_controller(args):
    logger.info("launch_controller")
    controller = Controller(args.dispatch_method)
    controller_module.controller = controller
    uvicorn.run(controller_app, host=args.host, port=args.controller_port, log_level=args.log_level)


# launch the model worker
def launch_model_worker(args):
    logger.info("launch_model_worker")
    gptq_config = GptqConfig(
        ckpt=args.gptq_ckpt or args.model_path,
        wbits=args.gptq_wbits,
        groupsize=args.gptq_groupsize,
        act_order=args.gptq_act_order,
    )

    worker = ModelWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.no_register,
        args.model_path,
        args.model_names,
        args.device,
        args.num_gpus,
        args.max_gpu_memory,
        args.load_8bit,
        args.cpu_offloading,
        gptq_config,
    )
    model_worker_module.worker = worker
    uvicorn.run(worker_app, host=args.host, port=args.worker_port, log_level=args.log_level)


# launch the RESTful API server
def launch_restful_api_server(args):
    logger.info("launch_restful_api_server")

    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=args.allowed_origins,
        allow_credentials=args.allow_credentials,
        allow_methods=args.allowed_methods,
        allow_headers=args.allowed_headers,
    )
    app_settings.controller_address = args.controller_address
    app_settings.api_keys = args.api_keys


    uvicorn.run(api_app, host=args.host, port=args.api_port, log_level=args.log_level)


if __name__ == '__main__':

    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser()

    # 添加命令行参数
    parser.add_argument('--launch', choices=['controller', 'worker', 'api'], default='default')
    parser.add_argument('--log_level', choices=['debug', 'info', 'warning', 'error'], default='info')

    # controller
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--controller_port", type=int, default=21001)
    parser.add_argument(
        "--dispatch-method",
        type=str,
        choices=["lottery", "shortest_queue"],
        default="shortest_queue",
    )

    # model worker
    parser.add_argument("--worker_port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
    parser.add_argument(
        "--controller-address", type=str, default="http://localhost:21001"
    )
    add_model_args(parser)
    parser.add_argument(
        "--model-names",
        type=lambda s: s.split(","),
        help="Optional display comma separated names",
    )
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")

    # openai api
    parser.add_argument("--api_port", type=int, default=8000, help="port number")
    parser.add_argument(
        "--allow-credentials", action="store_true", help="allow credentials"
    )
    parser.add_argument(
        "--allowed-origins", type=json.loads, default=["*"], help="allowed origins"
    )
    parser.add_argument(
        "--allowed-methods", type=json.loads, default=["*"], help="allowed methods"
    )
    parser.add_argument(
        "--allowed-headers", type=json.loads, default=["*"], help="allowed headers"
    )
    parser.add_argument(
        "--api-keys",
        type=lambda s: s.split(","),
        help="Optional list of comma separated API keys",
    )

    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    if args.launch == 'controller':
        launch_controller(args)
    elif args.launch == 'worker':
        launch_model_worker(args)
    elif args.launch == 'api':
        launch_restful_api_server(args)
    else:
        print("exit")
        exit(1)
