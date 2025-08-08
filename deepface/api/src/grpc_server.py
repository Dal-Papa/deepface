import argparse
from concurrent import futures

import grpc

from deepface import DeepFace
from deepface.commons.logger import Logger

# Import your generated gRPC module and service implementation for the unified service
import deepface.api.proto.deepface_pb2_grpc as deepface_grpc

logger = Logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, default=50051, help="Port of serving api")
    parser.add_argument("-w", "--workers", type=int, default=10, help="Maximum worker threads")
    args = parser.parse_args()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=args.workers))
    # Register the unified DeepFaceService
    deepface_grpc.add_DeepFaceServiceServicer_to_server(deepface_grpc.DeepFaceService(), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    # Start the gRPC server
    logger.info(f"gRPC server running on port {args.port}")
    logger.info(f"Welcome to DeepFace API v{DeepFace.__version__}!")
    server.wait_for_termination()
