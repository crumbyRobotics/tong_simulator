import zmq
import numpy as np
from typing import Tuple, Dict, Any, List


class SocketManager:
    def __init__(self, ip_addr: str = "localhost:5555"):
        ctx = zmq.Context()
        # NOTE Simulator is in REQEST mode: User programs should be implemented in REPLY mode
        self.socket = ctx.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip_addr}")

        # Resending each 2 sec if reply doesn't come
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.LINGER, 0)

    def receive(self) -> Tuple[list, np.ndarray, np.ndarray, np.ndarray, bool]:
        # Receiving data from the user program
        # NOTE: The user program should send a tuple of 5 elements
        input_tuple = self.socket.recv_multipart()
        if len(input_tuple) != 5:
            raise RuntimeError(f"Invalid size of tuple ({len(input_tuple)}) received from the user program. Expected 5.")

        # action command parameters
        cmd_params = np.frombuffer(input_tuple[0], dtype=np.float64).tolist()
        # action command for neck
        cmdN = np.frombuffer(input_tuple[1], dtype=np.float64).copy()
        # action command for left arm
        cmdL = np.frombuffer(input_tuple[2], dtype=np.float64).copy()
        # action command for right arm
        cmdR = np.frombuffer(input_tuple[3], dtype=np.float64).copy()
        reset = input_tuple[4] != b"\x00"  # reset flag

        return cmd_params, cmdN, cmdL, cmdR, reset

    def send(self, robot_data: tuple, world_data: Dict[str, Dict[str, np.ndarray]]):
        # Sending robot data to the user program
        posN, posL, posR, velN, velL, velR, fsL, fsR, SbSResultL, SbSResultR, SbSResultD = robot_data

        # Images shape
        LInfo = np.asarray(SbSResultL.shape, dtype=np.int32)
        RInfo = np.asarray(SbSResultR.shape, dtype=np.int32)
        DInfo = np.asarray(SbSResultD.shape, dtype=np.int32)

        # NOTE: zmq.SNDMORE is used to indicate that more frames will follow
        # NOTE: The last frame should not have zmq.SNDMORE flag
        self.socket.send(posN.astype(np.float64), zmq.SNDMORE)
        self.socket.send(posL.astype(np.float64), zmq.SNDMORE)
        self.socket.send(posR.astype(np.float64), zmq.SNDMORE)
        self.socket.send(velN.astype(np.float64), zmq.SNDMORE)
        self.socket.send(velL.astype(np.float64), zmq.SNDMORE)
        self.socket.send(velR.astype(np.float64), zmq.SNDMORE)
        self.socket.send(fsL.astype(np.float64), zmq.SNDMORE)
        self.socket.send(fsR.astype(np.float64), zmq.SNDMORE)
        self.socket.send(LInfo, zmq.SNDMORE)
        self.socket.send(SbSResultL.astype(np.uint8), zmq.SNDMORE)
        self.socket.send(RInfo, zmq.SNDMORE)
        self.socket.send(SbSResultR.astype(np.uint8), zmq.SNDMORE)
        self.socket.send(DInfo, zmq.SNDMORE)
        self.socket.send(SbSResultD.astype(np.uint16), zmq.SNDMORE)

        # Sending world data to the user program
        self.socket.send_pyobj(world_data, protocol=4)
