#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
wait_for_postgres() {
  echo "Waiting for PostgreSQL to be ready..."
  max_tries=30
  tries=0
  
  while [ $tries -lt $max_tries ]; do
    tries=$((tries + 1))
    
    # Try to connect to PostgreSQL
    python3 -c "
import sys, asyncio, asyncpg

async def test_connection():
    try:
        conn = await asyncpg.connect(
            host='$POSTGRES_HOST',
            port=$POSTGRES_PORT,
            user='$POSTGRES_USER',
            password='$POSTGRES_PASSWORD',
            database='$POSTGRES_DB'
        )
        await conn.close()
        return True
    except Exception as e:
        print(f'Error connecting to PostgreSQL: {e}')
        return False

if not asyncio.run(test_connection()):
    sys.exit(1)
" && break
    
    echo "PostgreSQL not ready yet, retrying in 3 seconds..."
    sleep 3
  done
  
  if [ $tries -eq $max_tries ]; then
    echo "Failed to connect to PostgreSQL after $max_tries attempts"
    exit 1
  fi
  
  echo "PostgreSQL is ready!"
}

# Initialize blockchain environment
initialize_blockchain() {
  echo "Initializing blockchain environment..."
  
  # Create necessary directories
  mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$LOG_DIR"
  
  # Generate initial config if it doesn't exist
  if [ ! -f "$CONFIG_DIR/config.json" ]; then
    echo "Generating initial configuration..."
    cat > "$CONFIG_DIR/config.json" << EOF
{
  "difficulty": ${INITIAL_DIFFICULTY:-4},
  "current_reward": ${INITIAL_REWARD:-50.0},
  "halving_interval": 210000,
  "mempool_max_size": 1000,
  "max_retries": 3,
  "sync_interval": 300,
  "use_gpu": ${USE_GPU:-false},
  "mining_workers": ${NUM_MINING_WORKERS:-2}
}
EOF
  fi
  
  # Initialize PostgreSQL schema if needed
  python3 -m enhanced_blockchain.db_init
  
  echo "Blockchain environment initialized!"
}

# Main entrypoint logic
if [ "$1" = "python3" ]; then
  # Wait for PostgreSQL when starting the blockchain node
  wait_for_postgres
  initialize_blockchain
fi

# Execute the given command
exec "$@"
