import argparse
from concurrent import futures

import grpc

from deepface import DeepFace
from deepface.commons.logger import Logger

# Import your generated gRPC modules and service implementations
import deepface.api.proto.analyze_pb2_grpc as analyze_grpc
import deepface.api.proto.represent_pb2_grpc as represent_grpc
import deepface.api.proto.verify_pb2_grpc as verify_grpc

logger = Logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=50051, help="Port of serving api")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Maximum worker threads")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    analyze_grpc.add_AnalyzeServiceServicer_to_server(analyze_grpc.AnalyzeService(), server)
    represent_grpc.add_RepresentServiceServicer_to_server(represent_grpc.RepresentService(), server)
    verify_grpc.add_VerifyServiceServicer_to_server(verify_grpc.VerifyService(), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    # Start the gRPC server
    logger.info(f"gRPC server running on port {args.port}")
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    server.wait_for_termination()
