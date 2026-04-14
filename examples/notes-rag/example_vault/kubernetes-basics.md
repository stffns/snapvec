---
tags: [kubernetes, infra]
---

# Kubernetes cheatsheet

## Pods, deployments, services — the 3-minute version

A **pod** is the smallest unit you deploy.  It wraps one container (usually)
plus shared storage, networking, and lifecycle.  You rarely create pods
directly — you create a **deployment**, and the deployment manages pods
via a replica set.

A **service** gives a stable network identity to a set of pods.  Without
one, your pods' IPs change every restart and nothing upstream can find
them reliably.  The service selector matches pod labels.

## Useful every-day commands

    kubectl get pods -A                  # all pods across namespaces
    kubectl logs -f <pod>                # tail logs
    kubectl describe pod <pod>           # events + container state
    kubectl exec -it <pod> -- /bin/sh    # open a shell inside the pod
    kubectl rollout restart deploy <d>   # rolling restart, no downtime

## When a pod is stuck

Most of the time it's one of: image pull backoff, a missing secret, an
unsatisfied resource request, or a readiness probe failing.  `describe`
tells you which, usually in the Events section at the bottom.
