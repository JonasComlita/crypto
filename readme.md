# Enhanced Blockchain Implementation

An optimized blockchain implementation with PostgreSQL storage, MessagePack serialization, and C++ acceleration.

## Features

- **C++ Acceleration**: Core cryptographic and compute-intensive operations implemented in C++
- **GPU Mining Support**: Optional CUDA acceleration for mining operations
- **PostgreSQL Storage**: Scalable database backend with optimized queries
- **MessagePack Serialization**: Efficient binary serialization format
- **Multiprocessing Mining**: Parallel mining across multiple CPU cores
- **Monitoring & Metrics**: Prometheus and Grafana integration

## System Requirements

- Python 3.8+
- PostgreSQL 15+
- CMake 3.10+
- C++ compiler (gcc 9+ or clang 10+)
- CUDA Toolkit 11+ (optional, for GPU mining)
- Docker and Docker Compose (for container deployment)

## Installation

### Using Docker (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/enhanced-blockchain.git
   cd enhanced-blockchain
   ```

2. Build and start the containers:
   ```bash
   # Without GPU support
   docker-compose up -d
   
   # With GPU support (requires NVIDIA Container Toolkit)
   WITH_GPU=true docker-compose up -d
   ```

3. View logs:
   ```bash
   docker-compose logs -f blockchain-node
   ```

### Manual Installation

1. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt install build-essential cmake libssl-dev libpq-dev python3-dev
   
   # macOS
   brew install cmake openssl postgresql python
   
   # Windows
   # Install Visual Studio with C++ support, CMake, and Python
   ```

2. Install PostgreSQL and create database:
   ```bash
   # Create database and user
   createuser -P blockchain
   createdb -O blockchain blockchain
   
   # Initialize database schema
   psql -U blockchain -d blockchain -f init-db.sql
   ```

3. Install Python package:
   ```bash
   # Without GPU support
   pip install -e .
   
   # With GPU support
   pip install -e .[gpu]
   ```

4. Initialize the database:
   ```bash
   python -m enhanced_blockchain.db_init
   ```

## Configuration

Create a `.env` file with the following variables:

```
# Database settings
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=blockchain
POSTGRES_PASSWORD=blockchain
POSTGRES_DB=blockchain

# Node settings
NODE_ID=node1
P2P_PORT=8001
API_PORT=8000
INITIAL_DIFFICULTY=4
INITIAL_REWARD=50.0
USE_GPU=false
NUM_MINING_WORKERS=2
```

## Running the Node

### Docker

The node starts automatically with Docker Compose. To restart:

```bash
docker-compose restart blockchain-node
```

### Manual

```bash
# Start the blockchain node
python -m enhanced_blockchain.main
```

## Monitoring

Access monitoring dashboards:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## Development

### Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=enhanced_blockchain
```

### Code Formatting

```bash
# Format code
black enhanced_blockchain tests

# Sort imports
isort enhanced_blockchain tests
```

### Building C++ Extension

```bash
# Build C++ extension
python setup.py build_ext --inplace
```

## API Documentation

The blockchain node exposes a REST API at `http://localhost:8000/api` and WebSocket at `ws://localhost:8000/ws`.

### Endpoints

- `GET /api/chain`: Get the full blockchain
- `GET /api/chain/latest`: Get the latest block
- `GET /api/transactions/{tx_id}`: Get transaction details
- `POST /api/transactions`: Submit a new transaction
- `GET /api/address/{address}`: Get address balance and transactions
- `GET /api/mining/stats`: Get mining statistics

## Architecture

The enhanced blockchain implementation follows a modular architecture:

1. **Core Components**
   - Enhanced blockchain with UTXO model
   - C++ acceleration for cryptographic operations
   - Multiprocessing mining with GPU support

2. **Storage Layer**
   - PostgreSQL database for blockchain data
   - MessagePack serialization for efficient storage
   - UTXO set management

3. **Network Layer**
   - P2P communication between nodes
   - Transaction and block propagation
   - Chain synchronization protocols

4. **API Layer**
   - REST API for external applications
   - WebSocket for real-time updates
   - JSON-RPC compatible endpoints

## License

MIT License
