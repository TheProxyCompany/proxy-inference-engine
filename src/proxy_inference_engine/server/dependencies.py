from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from proxy_inference_engine.server.ipc_dispatch import IPCState


async def get_ipc_state(request: Request) -> IPCState:
    """Dependency to get the shared IPC state."""
    ipc_state = getattr(request.app.state, "ipc_state", None)
    if not isinstance(ipc_state, IPCState):
        # Log error and raise appropriate HTTP exception
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server IPC infrastructure is not ready or initialized.",
        )
    return ipc_state


IPCStateDep = Annotated[IPCState, Depends(get_ipc_state)]
