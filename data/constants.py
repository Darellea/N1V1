"""
Constants for the data module.

This module contains all hardcoded values that have been extracted for better maintainability.
All values are documented with their purpose and units.
"""

# Cache Configuration
# Purpose: Time-to-live for cached data in seconds
# Units: seconds
CACHE_TTL = 3600

# Data Processing Configuration
# Purpose: Sample size for dataset hashing to detect changes
# Units: number of rows
HASH_SAMPLE_SIZE = 10000

# Retry Configuration
# Purpose: Maximum number of retry attempts for failed operations
# Units: number of attempts
MAX_RETRIES = 3

# Purpose: Delay between retry attempts
# Units: seconds
RETRY_DELAY = 5

# Rate Limiting Configuration
# Purpose: Default rate limit for API requests
# Units: requests per second
DEFAULT_RATE_LIMIT = 10

# Pagination Configuration
# Purpose: Maximum iterations to prevent infinite loops in pagination
# Units: number of iterations
MAX_PAGINATION_ITERATIONS = 1000

# Purpose: Threshold for using memory-efficient concatenation
# Units: number of DataFrames
MEMORY_EFFICIENT_THRESHOLD = 100

# Gap Handling Configuration
# Purpose: Default strategy for handling data gaps
# Options: 'forward_fill', 'interpolate', 'reject'
DEFAULT_GAP_HANDLING_STRATEGY = 'forward_fill'

# Cache Directory Configuration
# Purpose: Base directory for cache files
CACHE_BASE_DIR = 'data/cache'

# Historical Data Configuration
# Purpose: Base directory for historical data files
HISTORICAL_DATA_BASE_DIR = 'data/historical'
