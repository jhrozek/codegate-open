import argparse
import asyncio
import logging
import signal
from pathlib import Path

from codegate.pipeline.mcp import Manager

async def main(config_path: str):
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Create and initialize the manager
    manager = Manager(config_path)
    await manager.initialize()
    
    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(cleanup(manager)))
    
    try:
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await cleanup(manager)

async def cleanup(manager: Manager):
    """
    """
    logging.info("Shutting down MCP servers...")
    await manager.cleanup()
    # Stop the event loop
    asyncio.get_event_loop().stop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MCP servers from config")
    parser.add_argument(
        "config", 
        type=str,
        help="Path to MCP configuration file"
    )
    
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        exit(1)
        
    try:
        asyncio.run(main(str(config_path)))
    except KeyboardInterrupt:
        pass
