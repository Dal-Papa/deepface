import argparse
import threading
from concurrent import futures

import app
from flask import Flask

import grpc

# Import your generated gRPC modules and service implementations
import deepface.api.proto.analyze_pb2_grpc as analyze_grpc
import deepface.api.proto.represent_pb2_grpc as represent_grpc
import deepface.api.proto.verify_pb2_grpc as verify_grpc

# TODO: Implement these service classes
# from .grpc_services import AnalyzeService, RepresentService, VerifyService

def serve_grpc(port):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # Register your service implementations here

    analyze_grpc.add_AnalyzeServiceServicer_to_server(analyze_grpc.AnalyzeService(), server)
    represent_grpc.add_RepresentServiceServicer_to_server(represent_grpc.RepresentService(), server)
    verify_grpc.add_VerifyServiceServicer_to_server(verify_grpc.VerifyService(), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"gRPC server running on port {port}")
    server.wait_for_termination()

if __name__ == "__main__":
    deepface_app = app.create_app()
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=5000, help="Port of serving api")
    parser.add_argument("-g", "--grpc-port", type=int, default=50051, help="Port for gRPC server")
    args = parser.parse_args()

    # Start gRPC server in a separate thread
    grpc_thread = threading.Thread(target=serve_grpc, args=(args.grpc_port,), daemon=True)
    grpc_thread.start()

    # Start Flask HTTP server
    deepface_app.run(host="0.0.0.0", port=args.port)
