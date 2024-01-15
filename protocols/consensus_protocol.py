# consensus_protocol.py
import abc


class ConsensusProtocol(abc.ABC):
    @abc.abstractmethod
    def handle_message(self, message):
        pass
