# apex-mech

A high-performance autonomous combat bot written in Rust for the **Mech Royale** (16-mech battle royale) challenge.

`apex-mech` is designed to compete under strict arena constraints: sub-50 ms tick budget, single CPU core, 512 MB memory limit, unprivileged user execution, and a read-only filesystem.

---

## Architecture & Features

The bot is structured into modular components:

- **Protocol (`src/protocol.rs`)**: High-speed stdin/stdout JSON protocol handler. Employs background reading to prevent input buffer buildup and always act on the freshest match observation.
- **World & Spatial Modeling (`src/world.rs`)**: Parses `world.json` and 16-bit binary heightmaps (`terrain.i16`). Provides raymarching for 3D line-of-sight (LOS), prism collision checks, and height queries.
- **Navigation (`src/nav.rs`)**: Custom A* pathfinder respecting terrain slopes and step-up limits. Includes dynamic waypoint pruning, obstacle avoidance, stuck detection, and cover evaluation.
- **Belief State (`src/belief.rs`)**: Tracks enemy mechs, NPCs, crates, and power cubes. Uses dead-reckoning extrapolation for targets when visual contact is briefly lost, and adapts strategy based on match mode (FFA vs Practice).
- **Combat & Ballistics (`src/combat.rs`)**: Predictive targeting and aiming lead solutions for weapons (Laser, Lightning, Mortar, Machine Gun). Manages optimal weapon ranges, strafing, and evasive maneuvers.
- **Brain & Tactics (`src/main.rs`)**: High-level decision engine orchestrating looting priorities, dynamic upgrade tree choices, threat assessment, kiting, and cover seeking.

---

## Project Structure

```
apex-mech/
├── Cargo.toml          # Rust dependencies and release build profiles
├── Dockerfile          # Multi-stage container build targeting Debian Bookworm
├── mechfight-kit/      # Official tournament kit, arena maps, schemas, and checker
├── src/
│   ├── belief.rs       # World memory and target tracking
│   ├── combat.rs       # Ballistics and predictive aim logic
│   ├── main.rs         # Game loop and tactical decision state machine
│   ├── nav.rs          # A* pathfinding and movement engine
│   ├── protocol.rs     # JSON-lines protocol serializer/deserializer
│   ├── tactics.rs      # Tactical helpers
│   └── world.rs        # Terrain heightmap and 3D geometry engine
└── README.md
```

---

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/) (2021 edition, 1.80+)
- [Docker](https://www.docker.com/) (for containerized testing and submission)
- Python 3.8+ (for running the conformance checker)

### Build Locally

```sh
cargo build --release
```

The compiled binary will be placed at `target/release/apex-mech`.

### Build with Docker

```sh
docker build -t apex-mech .
```

---

## Testing & Conformance

Verify that the bot conforms to protocol and timing requirements using the bundled `mechfight-kit` checker.

### 1. Test Local Binary

```sh
python mechfight-kit/check/check.py -- cargo run --release
```

### 2. Test Container Image (Competition Environment)

Run the container using the exact resource limits enforced by the scorer:

```sh
docker run --rm -i \
  --network none \
  --cpus 1 \
  --memory 512m \
  --read-only \
  --tmpfs /tmp:rw,size=64m \
  --user 65534:65534 \
  -v "$PWD/mechfight-kit/arena:/arena:ro" \
  apex-mech
```

### 3. Check Container with `check.py`

```sh
python mechfight-kit/check/check.py --map-dir mechfight-kit/arena -- \
  docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/mechfight-kit/arena:/arena:ro" apex-mech
```

---

## Competition Specifications

| Metric | Limit / Setting |
| --- | --- |
| **Response Time** | 50 ms per tick |
| **CPU Allocation** | 1 Core |
| **Memory Limit** | 512 MB |
| **Network** | None (`--network none`) |
| **Filesystem** | Read-only (scratch space in `/tmp` up to 64 MB) |
| **User** | Unprivileged (`uid 65534`) |
| **Build Time** | Max 10 minutes on linux/amd64 |

---

## License

This project is licensed under the MIT License.

