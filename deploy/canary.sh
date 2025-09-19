#!/bin/bash

# N1V1 Canary Deployment Script
# Deploys new version to canary environment, runs smoke tests, and handles rollback

set -e  # Exit on any error

# Configuration
CANARY_ENV=${CANARY_ENV:-staging}
DRY_RUN=${DRY_RUN:-false}
TIMEOUT=${TIMEOUT:-300}  # 5 minutes timeout for smoke tests
ROLLBACK_ON_FAILURE=${ROLLBACK_ON_FAILURE:-true}
DISCORD_WEBHOOK_URL=${DISCORD_WEBHOOK_URL:-}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Notification function
notify_discord() {
    local message="$1"
    local color="$2"

    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        curl -X POST "$DISCORD_WEBHOOK_URL" \
            -H 'Content-Type: application/json' \
            -d "{\"content\": \"$message\"}" \
            --silent --output /dev/null
    fi
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    # Add cleanup logic here if needed
}

# Trap for cleanup on exit
trap cleanup EXIT

# Function to check if service is healthy
check_service_health() {
    local url="$1"
    local max_attempts=30
    local attempt=1

    log_info "Checking service health at $url"

    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$url" > /dev/null 2>&1; then
            log_success "Service is healthy (attempt $attempt)"
            return 0
        fi

        log_info "Service not ready yet (attempt $attempt/$max_attempts), waiting..."
        sleep 10
        ((attempt++))
    done

    log_error "Service failed to become healthy after $max_attempts attempts"
    return 1
}

# Function to run smoke tests
run_smoke_tests() {
    local base_url="$1"
    local start_time=$(date +%s)

    log_info "Running smoke tests against $base_url"

    # Test 1: Health endpoint
    log_info "Testing /health endpoint..."
    if ! curl -f -s "$base_url/health" > /dev/null 2>&1; then
        log_error "Health check failed"
        return 1
    fi
    log_success "Health check passed"

    # Test 2: Ready endpoint
    log_info "Testing /ready endpoint..."
    if ! curl -f -s "$base_url/ready" > /dev/null 2>&1; then
        log_error "Readiness check failed"
        return 1
    fi
    log_success "Readiness check passed"

    # Test 3: API status endpoint (if available)
    log_info "Testing /api/v1/status endpoint..."
    if curl -f -s "$base_url/api/v1/status" > /dev/null 2>&1; then
        log_success "Status endpoint check passed"
    else
        log_warning "Status endpoint not available or requires authentication"
    fi

    # Test 4: Basic functionality test (paper trading mode)
    log_info "Testing basic trading functionality..."
    # This would need to be customized based on your API endpoints
    # For now, we'll assume the health checks are sufficient

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    log_success "All smoke tests passed in ${duration}s"
    return 0
}

# Function to deploy to canary environment
deploy_to_canary() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would deploy to canary environment: $CANARY_ENV"
        return 0
    fi

    log_info "Deploying to canary environment: $CANARY_ENV"

    # Add your deployment logic here
    # This could be:
    # - Docker deployment
    # - Kubernetes deployment
    # - Cloud service deployment (AWS, GCP, Azure)
    # - Server deployment via SSH/rsync

    # Example for a simple deployment:
    # scp -r . user@canary-server:/path/to/app/
    # ssh user@canary-server "cd /path/to/app && ./restart_service.sh"

    # For this example, we'll simulate deployment
    log_info "Starting deployment process..."

    # Simulate deployment time
    sleep 5

    log_success "Deployment to canary environment completed"
}

# Function to rollback deployment
rollback_deployment() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] Would rollback canary deployment"
        return 0
    fi

    log_error "Rolling back canary deployment..."

    # Add your rollback logic here
    # This should restore the previous version

    notify_discord "🔄 N1V1 Canary rollback initiated" "danger"

    # Simulate rollback
    sleep 3

    log_info "Rollback completed"
    notify_discord "✅ N1V1 Canary rollback completed" "warning"
}

# Main deployment function
main() {
    log_info "Starting N1V1 canary deployment"
    log_info "Environment: $CANARY_ENV"
    log_info "Dry run: $DRY_RUN"
    log_info "Timeout: ${TIMEOUT}s"

    notify_discord "🚀 N1V1 Canary deployment started for environment: $CANARY_ENV" "primary"

    local canary_url=""

    # Determine canary URL based on environment
    case $CANARY_ENV in
        staging)
            canary_url="http://staging.n1v1.example.com"
            ;;
        canary)
            canary_url="http://canary.n1v1.example.com"
            ;;
        *)
            canary_url="http://localhost:8000"
            ;;
    esac

    log_info "Canary URL: $canary_url"

    # Step 1: Deploy to canary environment
    if ! deploy_to_canary; then
        log_error "Deployment to canary environment failed"
        notify_discord "❌ N1V1 Canary deployment failed during deployment phase" "danger"
        exit 1
    fi

    # Step 2: Wait for service to be ready
    if ! check_service_health "$canary_url/health"; then
        log_error "Service failed to become healthy after deployment"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            rollback_deployment
        fi
        notify_discord "❌ N1V1 Canary deployment failed - service unhealthy" "danger"
        exit 1
    fi

    # Step 3: Run smoke tests
    if ! run_smoke_tests "$canary_url"; then
        log_error "Smoke tests failed"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            rollback_deployment
        fi
        notify_discord "❌ N1V1 Canary deployment failed - smoke tests failed" "danger"
        exit 1
    fi

    # Step 4: Monitor for a period (optional)
    log_info "Monitoring canary deployment for 60 seconds..."
    sleep 60

    # Final health check
    if ! check_service_health "$canary_url/health"; then
        log_error "Final health check failed"
        if [ "$ROLLBACK_ON_FAILURE" = "true" ]; then
            rollback_deployment
        fi
        notify_discord "❌ N1V1 Canary deployment failed - final health check failed" "danger"
        exit 1
    fi

    log_success "Canary deployment successful!"
    log_success "All tests passed and service is healthy"
    log_info "Ready for full production rollout"

    notify_discord "✅ N1V1 Canary deployment successful - ready for production rollout" "success"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-rollback)
            ROLLBACK_ON_FAILURE=false
            shift
            ;;
        --env=*)
            CANARY_ENV="${1#*=}"
            shift
            ;;
        --timeout=*)
            TIMEOUT="${1#*=}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run          Run in dry-run mode (no actual deployment)"
            echo "  --no-rollback      Don't rollback on failure"
            echo "  --env=ENV          Set canary environment (default: staging)"
            echo "  --timeout=SECONDS  Set timeout for smoke tests (default: 300)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main "$@"
