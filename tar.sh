#!/bin/bash

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="experiments_${TIMESTAMP}"

TEMP_DIR="/tmp/${ARCHIVE_NAME}"
mkdir -p "$TEMP_DIR"

echo "=========================================="
echo "Packaging experiments from $(date)"
echo "=========================================="

echo "1. Copying checkpoints..."
if [ -d "./checkpoints" ]; then
    # Option A: Include only best.pt files (smaller)
    find ./checkpoints -name "best.pt" -exec cp --parents {} "$TEMP_DIR/" \;
    
    # Option B: Include entire checkpoint directories (uncomment if you want everything)
    # cp -r ../checkpoints "$TEMP_DIR/"
else
    echo "   Warning: ./checkpoints not found"
fi

echo "2. Copying results..."
if [ -d "./results" ]; then
    cp -r ./results "$TEMP_DIR/"
else
    echo "   Warning: ./results not found"
fi

echo "Creating archive..."
cd /tmp
tar -czf "${ARCHIVE_NAME}.tar.gz" "$ARCHIVE_NAME"
cd - > /dev/null

mv "/tmp/${ARCHIVE_NAME}.tar.gz" .
rm -rf "$TEMP_DIR"

echo "=========================================="
echo "Archive created: ${ARCHIVE_NAME}.tar.gz"
echo "Size: $(du -h ${ARCHIVE_NAME}.tar.gz | cut -f1)"
echo "=========================================="
echo ""
echo "To copy to remote machine:"
echo "  scp ${ARCHIVE_NAME}.tar.gz user@remote_host:~/"
echo ""
echo "To extract on remote:"
echo "  tar -xzf ${ARCHIVE_NAME}.tar.gz"
echo "  cp -r ${ARCHIVE_NAME}/checkpoints ."
echo "  cp -r ${ARCHIVE_NAME}/results ."
echo ""