#!/usr/bin/env python3
# Main entry point for Ultimate Vocal Remover
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main entry point"""
    try:
        # Always run in CLI mode
        logger.info('Running in CLI mode')
        from cli import CLI
        
        # Run CLI
        cli = CLI()
        sys.exit(cli.run())
            
    except KeyboardInterrupt:
        logger.info('Operation interrupted by user')
        sys.exit(1)
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
