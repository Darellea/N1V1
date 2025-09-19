# N1V1 Kubernetes Deployment Guide

This document provides comprehensive instructions for deploying N1V1 as a microservice architecture on Kubernetes with autoscaling, monitoring, and high availability.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Building Docker Images](#building-docker-images)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring and Observability](#monitoring-and-observability)
6. [Scaling Operations](#scaling-operations)
7. [Backup and Recovery](#backup-and-recovery)
8. [Troubleshooting](#troubleshooting)
9. [Security Considerations](#security-considerations)

## Architecture Overview

N1V1 is deployed as a microservice architecture with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │   Core Engine   │    │   ML Service    │
│   (Stateless)   │◄──►│  (Stateful)     │◄──►│  (StatefulSet)  │
│                 │    │                 │    │                 │
│ • REST API      │    │ • Trading Logic │    │ • Model Serving │
│ • Load Balance  │    │ • Order Mgmt    │    │ • Training      │
│ • Auto-scaling  │    │ • Risk Mgmt     │    │ • Persistence   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Monitoring     │
                    │  (Sidecar)      │
                    │                 │
                    │ • Prometheus    │
                    │ • Grafana       │
                    │ • AlertManager  │
                    └─────────────────┘
```

### Components

- **API Service**: Stateless REST API with horizontal scaling
- **Core Service**: Main trading engine with state management
- **ML Service**: Machine learning model serving with persistent storage
- **Monitoring**: Centralized observability stack

## Prerequisites

### System Requirements

- Kubernetes cluster (v1.19+)
- kubectl configured
- Docker registry access
- Helm 3.x (optional, for advanced deployments)
- NGINX Ingress Controller
- cert-manager (for TLS certificates)

### Cluster Resources

```yaml
# Minimum cluster requirements
nodes: 3
cpu: 8 cores total
memory: 16GB total
storage: 100GB total
```

### Required Tools

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl && sudo mv kubectl /usr/local/bin/

# Install Helm
curl https://get.helm.sh/helm-v3.9.0-linux-amd64.tar.gz -o helm.tar.gz
tar -zxvf helm.tar.gz && sudo mv linux-amd64/helm /usr/local/bin/

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

## Building Docker Images

### Build Process

1. **Clone the repository**
```bash
git clone https://github.com/Darellea/N1V1.git
cd N1V1
```

2. **Build all component images**
```bash
# Build API image
docker build -f deploy/Dockerfile.api -t n1v1-api:latest .

# Build Core image
docker build -f deploy/Dockerfile.core -t n1v1-core:latest .

# Build ML image
docker build -f deploy/Dockerfile.ml -t n1v1-ml:latest .

# Build Monitoring image
docker build -f deploy/Dockerfile.monitoring -t n1v1-monitoring:latest .
```

3. **Tag and push to registry**
```bash
# Tag images
docker tag n1v1-api:latest your-registry.com/n1v1-api:v1.0.0
docker tag n1v1-core:latest your-registry.com/n1v1-core:v1.0.0
docker tag n1v1-ml:latest your-registry.com/n1v1-ml:v1.0.0
docker tag n1v1-monitoring:latest your-registry.com/n1v1-monitoring:v1.0.0

# Push images
docker push your-registry.com/n1v1-api:v1.0.0
docker push your-registry.com/n1v1-core:v1.0.0
docker push your-registry.com/n1v1-ml:v1.0.0
docker push your-registry.com/n1v1-monitoring:v1.0.0
```

### Multi-Architecture Builds

```bash
# Build for multiple architectures
docker buildx build --platform linux/amd64,linux/arm64 \
  -f deploy/Dockerfile.api \
  -t your-registry.com/n1v1-api:v1.0.0 \
  --push .
```

## Kubernetes Deployment

### 1. Create Namespace

```bash
kubectl apply -f deploy/k8s/namespace.yaml
```

### 2. Deploy Configuration

```bash
# Deploy ConfigMap
kubectl apply -f deploy/k8s/configmap.yaml

# Verify configuration
kubectl get configmaps -n n1v1
```

### 3. Deploy Services

```bash
# Deploy all services
kubectl apply -f deploy/k8s/services.yaml

# Verify services
kubectl get services -n n1v1
```

### 4. Deploy Core Components

```bash
# Deploy API
kubectl apply -f deploy/k8s/deployment-api.yaml

# Deploy Core
kubectl apply -f deploy/k8s/deployment-core.yaml

# Deploy ML StatefulSet
kubectl apply -f deploy/k8s/statefulset-ml.yaml

# Verify deployments
kubectl get deployments -n n1v1
kubectl get statefulsets -n n1v1
```

### 5. Deploy Autoscaling

```bash
# Deploy HPA
kubectl apply -f deploy/k8s/hpa.yaml

# Verify HPA
kubectl get hpa -n n1v1
```

### 6. Deploy Ingress

```bash
# Deploy Ingress
kubectl apply -f deploy/k8s/ingress.yaml

# Verify Ingress
kubectl get ingress -n n1v1
```

### 7. Verify Deployment

```bash
# Check all resources
kubectl get all -n n1v1

# Check pod status
kubectl get pods -n n1v1 -o wide

# Check logs
kubectl logs -f deployment/n1v1-api -n n1v1
```

## Monitoring and Observability

### Prometheus Metrics

N1V1 exposes Prometheus metrics on the `/metrics` endpoint:

```bash
# Query API metrics
curl http://n1v1-api:8000/metrics

# Query Core metrics
curl http://n1v1-core:8000/metrics

# Query ML metrics
curl http://n1v1-ml:8080/metrics
```

### Health Checks

```bash
# API health check
curl http://n1v1-api:8000/health

# Readiness check
curl http://n1v1-api:8000/ready

# Deep health check
curl http://n1v1-api:8000/api/v1/status
```

### Logging

```bash
# View API logs
kubectl logs -f deployment/n1v1-api -n n1v1

# View Core logs
kubectl logs -f deployment/n1v1-core -n n1v1

# View ML logs
kubectl logs -f statefulset/n1v1-ml -n n1v1

# Centralized logging with labels
kubectl logs -l app=n1v1-api -n n1v1 --tail=100
```

## Scaling Operations

### Manual Scaling

```bash
# Scale API deployment
kubectl scale deployment n1v1-api --replicas=5 -n n1v1

# Scale Core deployment
kubectl scale deployment n1v1-core --replicas=2 -n n1v1

# Scale ML StatefulSet
kubectl scale statefulset n1v1-ml --replicas=2 -n n1v1
```

### Autoscaling Configuration

```bash
# View current HPA status
kubectl get hpa -n n1v1

# Describe HPA details
kubectl describe hpa n1v1-api-hpa -n n1v1

# Update HPA thresholds
kubectl patch hpa n1v1-api-hpa -n n1v1 \
  --type='json' \
  -p='[{"op": "replace", "path": "/spec/metrics/0/resource/target/averageUtilization", "value": 80}]'
```

### Scaling Best Practices

1. **Monitor resource usage** before scaling
2. **Scale gradually** to avoid resource contention
3. **Use HPA** for automatic scaling based on metrics
4. **Scale down** during low-traffic periods
5. **Monitor application performance** after scaling

## Backup and Recovery

### Database Backup

```bash
# Backup ConfigMap data
kubectl get configmap n1v1-config -n n1v1 -o yaml > backup-config.yaml

# Backup persistent volumes (ML models)
kubectl get pvc -n n1v1
kubectl describe pvc n1v1-ml-models-storage -n n1v1
```

### Application Backup

```bash
# Create backup of all resources
kubectl get all -n n1v1 -o yaml > n1v1-backup.yaml

# Backup with timestamps
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
kubectl get all -n n1v1 -o yaml > n1v1-backup-${TIMESTAMP}.yaml
```

### Recovery Procedures

```bash
# Restore from backup
kubectl apply -f n1v1-backup.yaml

# Rollback deployment
kubectl rollout undo deployment/n1v1-api -n n1v1

# Restore specific version
kubectl rollout undo deployment/n1v1-api --to-revision=2 -n n1v1
```

## Troubleshooting

### Common Issues

#### Pod Startup Failures

```bash
# Check pod events
kubectl describe pod <pod-name> -n n1v1

# Check pod logs
kubectl logs <pod-name> -n n1v1 --previous

# Check resource constraints
kubectl describe deployment n1v1-api -n n1v1
```

#### Service Connectivity Issues

```bash
# Test service DNS resolution
kubectl exec -it <pod-name> -n n1v1 -- nslookup n1v1-api

# Test service connectivity
kubectl exec -it <pod-name> -n n1v1 -- curl http://n1v1-api:8000/health

# Check service endpoints
kubectl get endpoints -n n1v1
```

#### Autoscaling Issues

```bash
# Check HPA status
kubectl describe hpa n1v1-api-hpa -n n1v1

# Check metrics server
kubectl get apiservices | grep metrics

# Verify resource requests/limits
kubectl describe deployment n1v1-api -n n1v1
```

#### Persistent Volume Issues

```bash
# Check PVC status
kubectl get pvc -n n1v1

# Check PV status
kubectl get pv

# Describe PVC for details
kubectl describe pvc n1v1-ml-models-storage -n n1v1
```

### Debug Commands

```bash
# Get cluster events
kubectl get events -n n1v1 --sort-by=.metadata.creationTimestamp

# Check node resources
kubectl describe nodes

# Check cluster capacity
kubectl get nodes -o custom-columns=NAME:.metadata.name,CPU:.status.capacity.cpu,MEMORY:.status.capacity.memory

# Debug with temporary pod
kubectl run debug-pod --image=busybox --rm -it --restart=Never -- sh
```

## Security Considerations

### Network Security

```yaml
# Network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: n1v1-network-policy
  namespace: n1v1
spec:
  podSelector:
    matchLabels:
      app: n1v1-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: n1v1-core
    ports:
    - protocol: TCP
      port: 8000
```

### Secrets Management

```bash
# Create secrets for sensitive data
kubectl create secret generic n1v1-secrets \
  --from-literal=api-key=your-api-key \
  --from-literal=secret-key=your-secret-key \
  --namespace n1v1

# Use secrets in deployments
env:
- name: API_KEY
  valueFrom:
    secretKeyRef:
      name: n1v1-secrets
      key: api-key
```

### RBAC Configuration

```yaml
# Service account for N1V1
apiVersion: v1
kind: ServiceAccount
metadata:
  name: n1v1-service-account
  namespace: n1v1

---
# Role for N1V1 operations
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: n1v1-role
  namespace: n1v1
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps"]
  verbs: ["get", "list", "watch"]

---
# Role binding
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: n1v1-role-binding
  namespace: n1v1
subjects:
- kind: ServiceAccount
  name: n1v1-service-account
roleRef:
  kind: Role
  name: n1v1-role
  apiGroup: rbac.authorization.k8s.io
```

## Performance Optimization

### Resource Optimization

```yaml
# Optimized resource requests/limits
resources:
  requests:
    memory: "256Mi"
    cpu: "200m"
  limits:
    memory: "512Mi"
    cpu: "500m"
```

### Horizontal Pod Autoscaling

```yaml
# Advanced HPA configuration
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: n1v1-api-hpa
spec:
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
```

### Monitoring Dashboards

Access Grafana dashboards for detailed performance metrics:

- **N1V1 API Performance**: Response times, throughput, error rates
- **Resource Usage**: CPU, memory, disk I/O
- **Trading Metrics**: Order volume, success rates, PnL
- **ML Model Performance**: Prediction accuracy, training times

## Maintenance Procedures

### Regular Maintenance Tasks

1. **Update Images**
```bash
# Update to latest version
kubectl set image deployment/n1v1-api api=n1v1-api:v1.1.0 -n n1v1

# Rolling update
kubectl rollout status deployment/n1v1-api -n n1v1
```

2. **Clean Up Resources**
```bash
# Remove completed jobs
kubectl delete jobs --field-selector status.successful=1 -n n1v1

# Clean up old replicasets
kubectl delete replicaset $(kubectl get replicaset -n n1v1 | grep '0' | awk '{print $1}') -n n1v1
```

3. **Backup Operations**
```bash
# Scheduled backups
kubectl create job backup-job --from=cronjob/n1v1-backup -n n1v1

# Verify backup integrity
kubectl logs job/backup-job -n n1v1
```

### Emergency Procedures

1. **Service Outage**
```bash
# Immediate scaling
kubectl scale deployment n1v1-api --replicas=10 -n n1v1

# Check service health
kubectl exec -it <pod-name> -n n1v1 -- curl http://localhost:8000/health
```

2. **Data Recovery**
```bash
# Restore from backup
kubectl apply -f backup/n1v1-backup-latest.yaml

# Verify data integrity
kubectl exec -it <pod-name> -n n1v1 -- check-data-integrity.sh
```

## Support and Resources

### Documentation Links

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Helm Documentation](https://helm.sh/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

### Community Resources

- [Kubernetes Slack Community](https://slack.k8s.io/)
- [Stack Overflow - Kubernetes](https://stackoverflow.com/questions/tagged/kubernetes)
- [Reddit - r/kubernetes](https://reddit.com/r/kubernetes)

### Professional Support

For enterprise support and consulting:

- Contact: support@n1v1.example.com
- Documentation: https://docs.n1v1.example.com
- Status Page: https://status.n1v1.example.com

---

*This deployment guide is maintained alongside the N1V1 codebase. Please update it when making changes to the deployment process.*
