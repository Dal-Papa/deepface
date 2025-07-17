#!/bin/bash
set -e

# Source and destination for .proto files
PROTO_SRC_DIR="deepface/api/proto"
PROTO_DST_DIR="deepface/api/proto"

# Locate grpc_tools _proto directory for well-known Google protos
GRPC_TOOLS_PROTO_DIR=$(python -c "import grpc_tools; import os; print(os.path.join(os.path.dirname(grpc_tools.__file__), '_proto'))")

echo "Generating protos..."
python -m grpc_tools.protoc \
  --proto_path="$PROTO_SRC_DIR" \
  --proto_path="$GRPC_TOOLS_PROTO_DIR" \
  --python_out="$PROTO_DST_DIR" \
  --grpc_python_out="$PROTO_DST_DIR" \
  --pyi_out="$PROTO_DST_DIR" \
  "$PROTO_SRC_DIR"/*.proto

echo "Fixing imports to use relative paths..."
# Fix `import X_pb2` → `from . import X_pb2`
find "$PROTO_DST_DIR" -name "*.py" -exec sed -i '' -E 's/^import (.+_pb2)/from . import \1/' {} \;

echo "Done."
