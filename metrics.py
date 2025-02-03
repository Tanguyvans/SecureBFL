import psutil
import GPUtil
from datetime import datetime


class MetricsTracker:
    def __init__(self, save_results):
        self.save_results = save_results
        self.energy_measurements = []
        self.protocol_communications = []  # Communications du protocole
        self.storage_communications = []   # Communications de stockage
        self.time_measurements = []
        self.start_time = None
        
    def start_tracking(self):
        self.start_time = datetime.now()
        
    def measure_power(self, round_num, phase):
        cpu_percent = psutil.cpu_percent()
        cpu_power = cpu_percent * 0.3  # Approximate watts based on CPU usage
        
        ram = psutil.virtual_memory()
        ram_power = (ram.used / ram.total) * 8  # Approximate RAM power consumption
        
        gpu_power = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_power = gpus[0].load * 150  # Approximate GPU power based on usage
        except:
            pass
            
        total_power = cpu_power + ram_power + gpu_power
        
        self.energy_measurements.append({
            'round': round_num,
            'phase': phase,
            'cpu_power': cpu_power,
            'ram_power': ram_power,
            'gpu_power': gpu_power,
            'total_power': total_power,
            'timestamp': datetime.now()
        })
        
    def record_protocol_communication(self, round_num, size_mb, comm_type):
        self.protocol_communications.append({
            'round': round_num,
            'size': size_mb,
            'type': comm_type,
            'timestamp': datetime.now()
        })
        
    def record_storage_communication(self, round_num, size_mb, operation):
        self.storage_communications.append({
            'round': round_num,
            'size': size_mb,
            'operation': operation,  # 'load' ou 'save'
            'timestamp': datetime.now()
        })
        
    def record_time(self, round_num, phase):
        self.time_measurements.append({
            'round': round_num,
            'phase': phase,
            'timestamp': datetime.now()
        })
        
    def save_metrics(self):
        duration = (datetime.now() - self.start_time).total_seconds()
        
        avg_cpu = sum(m['cpu_power'] for m in self.energy_measurements) / len(self.energy_measurements)
        avg_ram = sum(m['ram_power'] for m in self.energy_measurements) / len(self.energy_measurements)
        avg_gpu = sum(m['gpu_power'] for m in self.energy_measurements) / len(self.energy_measurements)
        
        total_power = avg_cpu + avg_ram + avg_gpu
        total_energy = (total_power * duration) / 1_000_000  # Convert to MJ
        
        with open(f"{self.save_results}/energy_metrics.txt", "w") as f:
            f.write("=== Energy Consumption Metrics ===\n\n")
            f.write(f"Total Duration: {duration:.2f} seconds\n")
            f.write(f"Average CPU Power: {avg_cpu:.2f} Watts\n")
            f.write(f"Average RAM Power: {avg_ram:.2f} Watts\n")
            f.write(f"Average GPU Power: {avg_gpu:.2f} Watts\n")
            f.write(f"Total Energy Consumption: {total_energy:.4f} Mega Joules\n\n")
            
            f.write("=== Detailed Measurements ===\n")
            for m in self.energy_measurements:
                f.write(f"Round {m['round']}, {m['phase']}: {m['total_power']:.2f}W "
                       f"(CPU: {m['cpu_power']:.2f}W, RAM: {m['ram_power']:.2f}W, GPU: {m['gpu_power']:.2f}W)\n")
                
        # Protocol Communication metrics
        protocol_by_type = {
            'node-node': 0,
            'node-client': 0,
            'client-client': 0,
            'client-node': 0
        }
        
        for m in self.protocol_communications:
            protocol_by_type[m['type']] += m['size']
            
        total_protocol = sum(protocol_by_type.values())
        
        # Storage Communication metrics
        total_storage = sum(m['size'] for m in self.storage_communications)
        
        with open(f"{self.save_results}/communication_metrics.txt", "w") as f:
            f.write("=== Communication Overhead Metrics ===\n\n")
            f.write("--- Protocol Communication ---\n")
            f.write(f"Total Protocol Communication: {total_protocol:.2f} MB\n")
            for comm_type, size in protocol_by_type.items():
                f.write(f"{comm_type}: {size:.2f} MB ({(size/total_protocol)*100:.1f}%)\n")
            
            f.write("\n--- Storage Communication ---\n")
            f.write(f"Total Storage Communication: {total_storage:.2f} MB\n")
            f.write("\nNote: Storage communication represents potential network overhead\n")
            f.write("if models were stored on a remote server instead of locally.\n")
            
            f.write("\n=== Detailed Measurements ===\n")
            f.write("\n-- Protocol Communications --\n")
            for m in self.protocol_communications:
                f.write(f"Round {m['round']}, {m['type']}: {m['size']:.2f} MB at {m['timestamp']}\n")
                
            f.write("\n-- Storage Communications --\n")
            for m in self.storage_communications:
                f.write(f"Round {m['round']}, {m['operation']}: {m['size']:.2f} MB at {m['timestamp']}\n")
        
        # Time metrics
        with open(f"{self.save_results}/time_metrics.txt", "w") as f:
            f.write("=== Time Metrics ===\n\n")
            f.write(f"Total Duration: {(datetime.now() - self.start_time).total_seconds():.2f} seconds\n\n")
            
            # Calculate phase durations
            f.write("=== Phase Durations ===\n")
            for i in range(len(self.time_measurements)-1):
                current = self.time_measurements[i]
                next_measurement = self.time_measurements[i+1]
                duration = (next_measurement['timestamp'] - current['timestamp']).total_seconds()
                f.write(f"Round {current['round']}, {current['phase']} -> {next_measurement['phase']}: {duration:.2f} seconds\n")
