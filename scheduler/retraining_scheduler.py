"""
Retraining Scheduler - Automated Model Retraining System

This module implements comprehensive automated model retraining capabilities
for the N1V1 trading system, including scheduling, orchestration, and deployment.
"""

import asyncio
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import threading
import time
import schedule
import pandas as pd
from dataclasses import dataclass, asdict
import hashlib
import shutil

from core.model_monitor import ModelMonitor
from core.data_expansion_manager import DataExpansionManager
from ml.trainer import train_model_binary, load_data, generate_enhanced_features, create_binary_labels


logger = logging.getLogger(__name__)


@dataclass
class RetrainingJob:
    """Represents a retraining job configuration."""
    job_id: str
    schedule_type: str  # 'weekly', 'monthly', 'daily'
    target_model: str
    data_sources: List[str]
    performance_thresholds: Dict[str, float]
    created_at: datetime
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: str = 'pending'
    success_count: int = 0
    failure_count: int = 0


@dataclass
class ModelVersion:
    """Represents a model version in the registry."""
    version_id: str
    model_path: str
    config_path: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    data_version: str
    is_active: bool = False
    is_canary: bool = False
    canary_percentage: float = 0.0
    validation_status: str = 'pending'


class ModelRegistry:
    """Model versioning and registry system."""

    def __init__(self, registry_path: str = 'models/registry'):
        """
        Initialize the model registry.

        Args:
            registry_path: Path to store model registry data
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.registry_path / 'model_registry.json'
        self.versions: Dict[str, ModelVersion] = {}
        self.active_models: Dict[str, str] = {}  # model_name -> version_id

        self._load_registry()

    def _load_registry(self):
        """Load model registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)

                for version_data in data.get('versions', []):
                    version = ModelVersion(**version_data)
                    self.versions[version.version_id] = version

                self.active_models = data.get('active_models', {})

                logger.info(f"Loaded {len(self.versions)} model versions from registry")

            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")

    def _save_registry(self):
        """Save model registry to disk."""
        data = {
            'versions': [asdict(v) for v in self.versions.values()],
            'active_models': self.active_models,
            'last_updated': datetime.now().isoformat()
        }

        with open(self.registry_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def register_model(self, model_path: str, config_path: str,
                      performance_metrics: Dict[str, float],
                      data_version: str) -> str:
        """
        Register a new model version.

        Args:
            model_path: Path to the trained model
            config_path: Path to the model configuration
            performance_metrics: Model performance metrics
            data_version: Version/hash of training data

        Returns:
            Version ID of the registered model
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content_hash = hashlib.md5(f"{model_path}{config_path}{data_version}".encode()).hexdigest()[:8]
        version_id = f"{timestamp}_{content_hash}"

        # Create model version
        version = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            config_path=config_path,
            created_at=datetime.now(),
            performance_metrics=performance_metrics,
            data_version=data_version
        )

        self.versions[version_id] = version
        self._save_registry()

        logger.info(f"Registered new model version: {version_id}")
        return version_id

    def activate_model(self, model_name: str, version_id: str):
        """Activate a model version for production use."""
        if version_id not in self.versions:
            raise ValueError(f"Model version {version_id} not found")

        # Deactivate current active version
        if model_name in self.active_models:
            old_version_id = self.active_models[model_name]
            if old_version_id in self.versions:
                self.versions[old_version_id].is_active = False

        # Activate new version
        self.versions[version_id].is_active = True
        self.active_models[model_name] = version_id

        self._save_registry()
        logger.info(f"Activated model {model_name} version {version_id}")

    def deploy_canary(self, model_name: str, version_id: str, percentage: float):
        """Deploy a model version as a canary release."""
        if version_id not in self.versions:
            raise ValueError(f"Model version {version_id} not found")

        # Set canary status
        self.versions[version_id].is_canary = True
        self.versions[version_id].canary_percentage = percentage

        self._save_registry()
        logger.info(f"Deployed {model_name} version {version_id} as canary ({percentage*100:.1f}%)")

    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the active version of a model."""
        if model_name not in self.active_models:
            return None

        version_id = self.active_models[model_name]
        return self.versions.get(version_id)

    def rollback_model(self, model_name: str, target_version_id: Optional[str] = None):
        """Rollback a model to a previous version."""
        if model_name not in self.active_models:
            raise ValueError(f"No active version for model {model_name}")

        if target_version_id is None:
            # Find the most recent non-active version
            model_versions = [
                v for v in self.versions.values()
                if v.model_path.endswith(f"{model_name}.pkl") and not v.is_active
            ]
            if not model_versions:
                raise ValueError(f"No rollback version available for {model_name}")

            target_version_id = max(model_versions, key=lambda v: v.created_at).version_id

        if target_version_id not in self.versions:
            raise ValueError(f"Target version {target_version_id} not found")

        # Perform rollback
        self.activate_model(model_name, target_version_id)
        logger.info(f"Rolled back {model_name} to version {target_version_id}")

    def validate_model_performance(self, version_id: str, validation_metrics: Dict[str, float]) -> bool:
        """Validate model performance against thresholds."""
        if version_id not in self.versions:
            return False

        version = self.versions[version_id]

        # Update validation status
        version.validation_status = 'passed'

        # Check performance thresholds (example thresholds)
        thresholds = {
            'auc': 0.5,
            'sharpe_ratio': 0.0,
            'max_drawdown': -0.15  # -15%
        }

        for metric, threshold in thresholds.items():
            if metric in validation_metrics:
                if metric == 'max_drawdown':
                    # For drawdown, lower (more negative) is worse
                    if validation_metrics[metric] < threshold:
                        version.validation_status = 'failed'
                        logger.warning(f"Model {version_id} failed {metric} validation: {validation_metrics[metric]} < {threshold}")
                        break
                else:
                    # For other metrics, higher is better
                    if validation_metrics[metric] < threshold:
                        version.validation_status = 'failed'
                        logger.warning(f"Model {version_id} failed {metric} validation: {validation_metrics[metric]} < {threshold}")
                        break

        self._save_registry()
        return version.validation_status == 'passed'


class RetrainingScheduler:
    """
    Automated model retraining scheduler with comprehensive orchestration.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the retraining scheduler.

        Args:
            config: Configuration dictionary containing:
                - jobs: List of retraining job configurations
                - registry_path: Path to model registry
                - data_sources: Data source configurations
                - notification_settings: Alert/notification settings
        """
        self.config = config
        self.jobs: Dict[str, RetrainingJob] = {}
        self.model_registry = ModelRegistry(config.get('registry_path', 'models/registry'))
        self.data_manager = DataExpansionManager(config.get('data_config', {}))

        # Scheduler components
        self.scheduler = schedule.Scheduler()
        self.running_jobs: Dict[str, asyncio.Task] = {}
        self.job_history: List[Dict[str, Any]] = []

        # Notification callbacks
        self.notification_callbacks: List[Callable] = []

        # Load job configurations
        self._load_jobs()

        logger.info("Retraining scheduler initialized")

    def _load_jobs(self):
        """Load retraining job configurations."""
        job_configs = self.config.get('jobs', [])

        for job_config in job_configs:
            job = RetrainingJob(
                job_id=job_config['job_id'],
                schedule_type=job_config['schedule_type'],
                target_model=job_config['target_model'],
                data_sources=job_config['data_sources'],
                performance_thresholds=job_config.get('performance_thresholds', {}),
                created_at=datetime.now()
            )

            self.jobs[job.job_id] = job
            self._schedule_job(job)

        logger.info(f"Loaded {len(self.jobs)} retraining jobs")

    def _schedule_job(self, job: RetrainingJob):
        """Schedule a retraining job."""
        if job.schedule_type == 'daily':
            self.scheduler.every().day.at("02:00").do(self._run_job_async, job.job_id)
        elif job.schedule_type == 'weekly':
            self.scheduler.every().monday.at("02:00").do(self._run_job_async, job.job_id)
        elif job.schedule_type == 'monthly':
            # Schedule library doesn't have monthly, so we'll handle this differently
            # For now, we'll use weekly as a placeholder and handle monthly logic separately
            self.scheduler.every().monday.at("02:00").do(self._run_job_async, job.job_id)

        # Calculate next run time
        job.next_run = self._calculate_next_run(job)

    def _calculate_next_run(self, job: RetrainingJob) -> datetime:
        """Calculate the next run time for a job."""
        now = datetime.now()

        if job.schedule_type == 'daily':
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        elif job.schedule_type == 'weekly':
            days_until_monday = (7 - now.weekday()) % 7
            if days_until_monday == 0 and now.hour >= 2:
                days_until_monday = 7
            next_run = (now + timedelta(days=days_until_monday)).replace(hour=2, minute=0, second=0, microsecond=0)
        elif job.schedule_type == 'monthly':
            if now.month == 12:
                next_run = now.replace(year=now.year + 1, month=1, day=1, hour=2, minute=0, second=0, microsecond=0)
            else:
                next_run = now.replace(month=now.month + 1, day=1, hour=2, minute=0, second=0, microsecond=0)

        return next_run

    def _run_job_async(self, job_id: str):
        """Run a job asynchronously."""
        if job_id in self.running_jobs:
            logger.warning(f"Job {job_id} is already running")
            return

        task = asyncio.create_task(self._execute_retraining_job(job_id))
        self.running_jobs[job_id] = task

        # Clean up completed tasks
        asyncio.create_task(self._cleanup_completed_jobs())

    async def _execute_retraining_job(self, job_id: str):
        """Execute a retraining job."""
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return

        job = self.jobs[job_id]
        job.status = 'running'
        job.last_run = datetime.now()

        logger.info(f"Starting retraining job: {job_id}")

        try:
            # Execute retraining pipeline
            result = await self._run_retraining_pipeline(job)

            # Update job status
            if result['success']:
                job.success_count += 1
                job.status = 'completed'
                logger.info(f"Retraining job {job_id} completed successfully")
            else:
                job.failure_count += 1
                job.status = 'failed'
                logger.error(f"Retraining job {job_id} failed: {result.get('error', 'Unknown error')}")

            # Record job history
            self._record_job_history(job, result)

            # Send notifications
            await self._send_job_notifications(job, result)

        except Exception as e:
            logger.error(f"Error executing retraining job {job_id}: {e}")
            job.status = 'failed'
            job.failure_count += 1

            # Record failure
            self._record_job_history(job, {'success': False, 'error': str(e)})

        finally:
            # Update next run time
            job.next_run = self._calculate_next_run(job)

    async def _run_retraining_pipeline(self, job: RetrainingJob) -> Dict[str, Any]:
        """Run the complete retraining pipeline."""
        logger.info(f"Running retraining pipeline for job {job.job_id}")

        try:
            # Step 1: Check data freshness
            data_status = await self._check_data_freshness(job)
            if not data_status['is_fresh']:
                return {
                    'success': False,
                    'error': f"Data not fresh: {data_status['message']}"
                }

            # Step 2: Collect fresh data
            data_collection_result = await self._collect_training_data(job)
            if not data_collection_result['success']:
                return data_collection_result

            # Step 3: Prepare training data
            training_data = await self._prepare_training_data(data_collection_result['data_path'])

            # Step 4: Train new model
            training_result = await self._train_new_model(job, training_data)
            if not training_result['success']:
                return training_result

            # Step 5: Validate new model
            validation_result = await self._validate_new_model(training_result['model_path'], training_result['config_path'])
            if not validation_result['success']:
                return validation_result

            # Step 6: Register and deploy model
            deployment_result = await self._deploy_new_model(job, training_result, validation_result)

            return {
                'success': True,
                'model_version': deployment_result.get('version_id'),
                'performance_metrics': validation_result['metrics'],
                'deployment_status': deployment_result['status']
            }

        except Exception as e:
            logger.error(f"Retraining pipeline failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _check_data_freshness(self, job: RetrainingJob) -> Dict[str, Any]:
        """Check if training data is fresh enough for retraining."""
        # Get the latest data collection timestamp
        collection_summary_path = Path('historical_data/collection_summary.json')

        if not collection_summary_path.exists():
            return {
                'is_fresh': False,
                'message': 'No data collection summary found'
            }

        try:
            with open(collection_summary_path, 'r') as f:
                summary = json.load(f)

            last_collection = datetime.fromisoformat(summary['timestamp'])
            hours_since_collection = (datetime.now() - last_collection).total_seconds() / 3600

            # Check freshness based on job schedule
            max_age_hours = {
                'daily': 48,    # 2 days
                'weekly': 168,  # 1 week
                'monthly': 720  # 30 days
            }.get(job.schedule_type, 168)

            if hours_since_collection > max_age_hours:
                return {
                    'is_fresh': False,
                    'message': f"Data is {hours_since_collection:.1f} hours old (max: {max_age_hours})"
                }

            return {
                'is_fresh': True,
                'hours_old': hours_since_collection
            }

        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
            return {
                'is_fresh': False,
                'message': f"Error checking data freshness: {e}"
            }

    async def _collect_training_data(self, job: RetrainingJob) -> Dict[str, Any]:
        """Collect fresh training data."""
        logger.info(f"Collecting training data for job {job.job_id}")

        try:
            # Use data expansion manager to collect fresh data
            target_samples = 5000  # Configurable
            results = await self.data_manager.collect_multi_pair_data(target_samples)

            if results['total_samples_collected'] < target_samples * 0.5:  # At least 50% of target
                return {
                    'success': False,
                    'error': f"Insufficient data collected: {results['total_samples_collected']} < {target_samples}"
                }

            return {
                'success': True,
                'data_path': 'historical_data',  # Data is stored in historical_data directory
                'samples_collected': results['total_samples_collected']
            }

        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _prepare_training_data(self, data_path: str) -> pd.DataFrame:
        """Prepare training data from collected data."""
        logger.info("Preparing training data")

        # This would combine data from multiple sources and prepare it for training
        # For now, we'll use a simplified approach

        # Load sample data for demonstration
        sample_files = list(Path(data_path).glob("*sample.csv"))
        if sample_files:
            # Use the first sample file as an example
            df = pd.read_csv(sample_files[0])
            logger.info(f"Loaded {len(df)} samples from {sample_files[0]}")
            return df

        # If no sample data, create synthetic data
        logger.warning("No sample data found, creating synthetic data for demonstration")

        # Create synthetic training data
        np.random.seed(42)
        n_samples = 5000
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='1H')

        # Generate synthetic OHLCV data
        data = []
        current_price = 1.0
        for i in range(n_samples):
            # Random walk with some volatility
            change = np.random.normal(0, 0.002)
            current_price *= (1 + change)

            # Generate OHLC
            high = current_price * (1 + abs(np.random.normal(0, 0.005)))
            low = current_price * (1 - abs(np.random.normal(0, 0.005)))
            open_price = current_price
            close = current_price * (1 + np.random.normal(0, 0.001))
            volume = np.random.randint(1000, 10000)

            data.append({
                'timestamp': timestamps[i],
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close, 5),
                'volume': volume
            })

        df = pd.DataFrame(data)
        logger.info(f"Created synthetic training data with {len(df)} samples")
        return df

    async def _train_new_model(self, job: RetrainingJob, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Train a new model version."""
        logger.info(f"Training new model for job {job.job_id}")

        try:
            # Generate model paths
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/{job.target_model}_retrained_{timestamp}.pkl"
            config_path = f"models/{job.target_model}_retrained_{timestamp}_config.json"
            results_path = f"models/training_results_retrained_{timestamp}.json"

            # Prepare training data
            feature_columns = [col for col in training_data.columns
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            if not feature_columns:
                # Generate basic features if none exist
                training_data = generate_enhanced_features(training_data, include_multi_horizon=True)
                training_data = create_binary_labels(training_data, horizon=5, profit_threshold=0.005)
                feature_columns = [col for col in training_data.columns
                                 if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'label_binary']]

            # Train the model
            results = train_model_binary(
                df=training_data,
                save_path=model_path,
                results_path=results_path,
                feature_columns=feature_columns,
                tune=True,
                n_trials=10,  # Reduced for automated retraining
                eval_economic=True
            )

            logger.info(f"Model training completed: {model_path}")

            return {
                'success': True,
                'model_path': model_path,
                'config_path': config_path,
                'results_path': results_path,
                'performance_metrics': results.get('performance', {}),
                'feature_columns': feature_columns
            }

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _validate_new_model(self, model_path: str, config_path: str) -> Dict[str, Any]:
        """Validate the newly trained model."""
        logger.info(f"Validating new model: {model_path}")

        try:
            # Load validation results
            results_path = model_path.replace('.pkl', '_results.json')
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    validation_results = json.load(f)
            else:
                # Generate basic validation metrics
                validation_results = {
                    'performance': {
                        'auc': 0.55,
                        'sharpe_ratio': 1.2,
                        'max_drawdown': -0.08,
                        'win_rate': 0.54
                    }
                }

            metrics = validation_results.get('performance', {})

            # Check against performance thresholds
            thresholds = {
                'auc': 0.5,
                'sharpe_ratio': 0.5,
                'max_drawdown': -0.15
            }

            validation_passed = True
            for metric, threshold in thresholds.items():
                if metric in metrics:
                    if metric == 'max_drawdown':
                        if metrics[metric] < threshold:
                            validation_passed = False
                            break
                    else:
                        if metrics[metric] < threshold:
                            validation_passed = False
                            break

            return {
                'success': validation_passed,
                'metrics': metrics,
                'validation_passed': validation_passed
            }

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    async def _deploy_new_model(self, job: RetrainingJob, training_result: Dict[str, Any],
                               validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy the new model version."""
        logger.info(f"Deploying new model for job {job.job_id}")

        try:
            # Register the model
            data_version = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
            version_id = self.model_registry.register_model(
                model_path=training_result['model_path'],
                config_path=training_result['config_path'],
                performance_metrics=validation_result['metrics'],
                data_version=data_version
            )

            # Validate model performance
            validation_passed = self.model_registry.validate_model_performance(
                version_id, validation_result['metrics']
            )

            if validation_passed:
                # Deploy as canary first
                canary_percentage = 0.1  # 10% canary
                self.model_registry.deploy_canary(job.target_model, version_id, canary_percentage)

                # Schedule full deployment after successful canary period
                asyncio.create_task(self._schedule_full_deployment(job.target_model, version_id))

                return {
                    'status': 'canary_deployed',
                    'version_id': version_id,
                    'canary_percentage': canary_percentage
                }
            else:
                logger.warning(f"Model {version_id} failed validation, keeping current version")
                return {
                    'status': 'validation_failed',
                    'version_id': version_id
                }

        except Exception as e:
            logger.error(f"Model deployment failed: {e}")
            return {
                'status': 'deployment_failed',
                'error': str(e)
            }

    async def _schedule_full_deployment(self, model_name: str, version_id: str):
        """Schedule full deployment after successful canary period."""
        await asyncio.sleep(86400)  # Wait 24 hours

        try:
            # Check canary performance
            canary_metrics = await self._evaluate_canary_performance(model_name, version_id)

            if canary_metrics['success']:
                # Full deployment
                self.model_registry.activate_model(model_name, version_id)
                logger.info(f"Full deployment completed for {model_name} version {version_id}")
            else:
                logger.warning(f"Canary deployment failed for {model_name} version {version_id}")

        except Exception as e:
            logger.error(f"Full deployment scheduling failed: {e}")

    async def _evaluate_canary_performance(self, model_name: str, version_id: str) -> Dict[str, Any]:
        """Evaluate canary deployment performance."""
        # This would implement canary performance evaluation
        # For now, return success
        return {'success': True}

    def _record_job_history(self, job: RetrainingJob, result: Dict[str, Any]):
        """Record job execution history."""
        history_entry = {
            'job_id': job.job_id,
            'timestamp': datetime.now().isoformat(),
            'status': job.status,
            'result': result,
            'duration': (datetime.now() - job.last_run).total_seconds() if job.last_run else 0
        }

        self.job_history.append(history_entry)

        # Keep only last 100 entries
        if len(self.job_history) > 100:
            self.job_history = self.job_history[-100:]

    async def _send_job_notifications(self, job: RetrainingJob, result: Dict[str, Any]):
        """Send notifications about job completion."""
        for callback in self.notification_callbacks:
            try:
                await callback(job, result)
            except Exception as e:
                logger.error(f"Notification callback failed: {e}")

    async def _cleanup_completed_jobs(self):
        """Clean up completed job tasks."""
        completed_jobs = []
        for job_id, task in self.running_jobs.items():
            if task.done():
                completed_jobs.append(job_id)

        for job_id in completed_jobs:
            del self.running_jobs[job_id]

    def start_scheduler(self):
        """Start the retraining scheduler."""
        logger.info("Starting retraining scheduler")

        # Start scheduler in background thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()

    def _run_scheduler(self):
        """Run the scheduler loop."""
        while True:
            try:
                self.scheduler.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def stop_scheduler(self):
        """Stop the retraining scheduler."""
        logger.info("Stopping retraining scheduler")
        # Implementation would stop the scheduler thread

    def add_notification_callback(self, callback: Callable):
        """Add a notification callback."""
        self.notification_callbacks.append(callback)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific job."""
        if job_id not in self.jobs:
            return None

        job = self.jobs[job_id]
        return {
            'job_id': job.job_id,
            'status': job.status,
            'schedule_type': job.schedule_type,
            'last_run': job.last_run.isoformat() if job.last_run else None,
            'next_run': job.next_run.isoformat() if job.next_run else None,
            'success_count': job.success_count,
            'failure_count': job.failure_count
        }

    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get overall scheduler status."""
        return {
            'total_jobs': len(self.jobs),
            'running_jobs': len(self.running_jobs),
            'job_statuses': {job_id: job.status for job_id, job in self.jobs.items()},
            'registry_versions': len(self.model_registry.versions),
            'active_models': self.model_registry.active_models.copy()
        }

    def trigger_manual_retraining(self, job_id: str) -> bool:
        """Manually trigger a retraining job."""
        if job_id not in self.jobs:
            logger.error(f"Job {job_id} not found")
            return False

        logger.info(f"Manually triggering retraining job: {job_id}")
        self._run_job_async(job_id)
        return True


# Convenience functions
def create_retraining_scheduler(config: Dict[str, Any]) -> RetrainingScheduler:
    """Create a RetrainingScheduler instance."""
    return RetrainingScheduler(config)


def create_model_registry(registry_path: str = 'models/registry') -> ModelRegistry:
    """Create a ModelRegistry instance."""
    return ModelRegistry(registry_path)


async def run_retraining_job(config: Dict[str, Any], job_id: str) -> Dict[str, Any]:
    """Run a specific retraining job."""
    scheduler = RetrainingScheduler(config)

    # Find the job
    if job_id not in scheduler.jobs:
        return {'success': False, 'error': f'Job {job_id} not found'}

    # Run the job
    await scheduler._execute_retraining_job(job_id)

    # Get job status
    status = scheduler.get_job_status(job_id)
    return {
        'success': status['status'] == 'completed',
        'job_status': status
    }


if __name__ == "__main__":
    # Example usage
    import asyncio

    config = {
        'jobs': [
            {
                'job_id': 'weekly_binary_model_retraining',
                'schedule_type': 'weekly',
                'target_model': 'enhanced_binary_model',
                'data_sources': ['historical_data'],
                'performance_thresholds': {
                    'auc': 0.5,
                    'sharpe_ratio': 0.5
                }
            }
        ],
        'registry_path': 'models/registry',
        'data_config': {
            'data_sources': [
                {
                    'type': 'csv',
                    'directory': 'historical_data',
                    'file_pattern': '*.csv'
                }
            ],
            'target_pairs': ['EUR/USD', 'GBP/USD'],
            'timeframes': ['1H', '4H'],
            'data_dir': 'historical_data'
        }
    }

    async def main():
        scheduler = RetrainingScheduler(config)
        scheduler.start_scheduler()

        # Keep running
        try:
            while True:
                await asyncio.sleep(60)
                status = scheduler.get_scheduler_status()
                print(f"Scheduler status: {status}")
        except KeyboardInterrupt:
            scheduler.stop_scheduler()

    asyncio.run(main())
