import torch.distributed as dist

__all__ = ["FutureResult", "SlavePipe", "SyncMaster"]


class FutureResult:
    """A thread-safe future implementation. Used only as one-to-one pipe."""

    def __init__(self):
        self._value = None

    def put(self, value):
        self._value = value

    def get(self):
        return self._value


class SlavePipe:
    def __init__(self, rank, sync_master) -> None:
        self.rank = rank
        self.sync_master = sync_master

    def run_slave(self, msg):
        return self.sync_master.run(self.rank, msg)


class SyncMaster:
    def __init__(self, master_callback):
        """
        Args:
            master_callback: a callback to be invoked after having collected messages from slave devices.
        """
        self.master_callback = master_callback

    def __getstate__(self):
        return {"master_callback": self.master_callback}

    def __setstate__(self, state):
        self.__init__(state["master_callback"])

    def register_slave(self, identifier):
        """
        Register a slave device.

        Args:
            identifier: an identifier, usually is the device id.

        Returns: a `SlavePipe` object which can be used to communicate with the master device.
        """
        return SlavePipe(identifier, self)

    def run(self, rank, msg):
        world_size = dist.get_world_size()
        is_master = rank == 0

        all_msgs = [None for _ in range(world_size)] if is_master else None
        dist.gather_object((rank, msg), object_gather_list=all_msgs, dst=0)

        results = None
        if is_master:
            results = self.master_callback(all_msgs)
            results.sort(key=lambda x: x[0])  # sort by rank
            results = [res for _, res in results]

        if is_master:
            dist.broadcast_object_list(results, src=0)
        else:
            results = [None for _ in range(world_size)]
            dist.broadcast_object_list(results, src=0)

        return results[rank]
