import psutil
import GPUtil
from datetime import datetime
import pyRAPL
import pynvml
import pcm
import os
import json

class MetricsTracker:
    def __init__(self, save_results):
        self.save_results = save_results
        self.energy_measurements = []
        self.protocol_communications = []  # Communications du protocole
        self.storage_communications = []   # Communications de stockage
        self.time_measurements = []
        self.start_time = None
        self.global_start_time = None
        self.global_start_energy = None
        
    def start_tracking(self):
        self.start_time = datetime.now()
        
    def measure_power(self, round_num, phase):
        """
        Mesure la consommation d'énergie avec fallback sur psutil si les outils spécialisés ne sont pas disponibles
        """
        # CPU Power - essai avec RAPL, fallback sur psutil
        try:
            pyRAPL.setup()
            cpu_measurement = pyRAPL.Measurement('cpu')
            cpu_measurement.begin()
            cpu_measurement.end()
            cpu_power = cpu_measurement.result.pkg[0]
        except:
            cpu_percent = psutil.cpu_percent()
            cpu_power = cpu_percent * 0.3  # Fallback sur l'estimation simple
        
        # RAM Power - fallback sur psutil si PCM n'est pas disponible
        try:
            ram_power = pcm.get_memory_power()
        except:
            ram = psutil.virtual_memory()
            ram_power = (ram.used / ram.total) * 8  # Fallback sur l'estimation simple
        
        # GPU Power - essai avec NVML, fallback sur GPUtil
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        except:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_power = gpus[0].load * 150
                else:
                    gpu_power = 0
            except:
                gpu_power = 0
            
        total_power = cpu_power + ram_power + gpu_power
        
        self.energy_measurements.append({
            'round': round_num,
            'phase': phase,
            'cpu_power': cpu_power,
            'ram_power': ram_power,
            'gpu_power': gpu_power,
            'total_power': total_power,
            'timestamp': datetime.now(),
            'measurement_type': {
                'cpu': 'RAPL' if 'pyRAPL' in locals() else 'psutil',
                'ram': 'PCM' if 'pcm' in locals() else 'psutil',
                'gpu': 'NVML' if 'pynvml' in locals() else 'GPUtil' if gpu_power > 0 else 'none'
            }
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

    def measure_global_power(self, phase):
        """
        Mesure globale de la consommation d'énergie
        """
        if phase == "start":
            self.global_start_time = datetime.now()
            self.measure_power('global', 'start')
            
        elif phase == "complete":
            self.measure_power('global', 'complete')
            duration = (datetime.now() - self.global_start_time).total_seconds()
            
            # Calculer la consommation totale
            global_measurements = [m for m in self.energy_measurements if m['round'] == 'global']
            if len(global_measurements) >= 2:
                start_measurement = global_measurements[0]
                end_measurement = global_measurements[-1]
                
                avg_power = {
                    'cpu': (start_measurement['cpu_power'] + end_measurement['cpu_power']) / 2,
                    'ram': (start_measurement['ram_power'] + end_measurement['ram_power']) / 2,
                    'gpu': (start_measurement['gpu_power'] + end_measurement['gpu_power']) / 2,
                    'total': (start_measurement['total_power'] + end_measurement['total_power']) / 2
                }
                
                # Calculer l'énergie totale en joules (Watts * secondes)
                total_energy_joules = avg_power['total'] * duration
                total_energy_megajoules = total_energy_joules / 1_000_000
                
                # Créer le rapport JSON
                total_energy_consumption = {
                    'duration_seconds': duration,
                    'start_time': start_measurement['timestamp'].isoformat(),
                    'end_time': end_measurement['timestamp'].isoformat(),
                    'average_power': avg_power,
                    'total_energy': {
                        'joules': total_energy_joules,
                        'megajoules': total_energy_megajoules
                    },
                    'measurement_type': start_measurement['measurement_type']
                }
                
                # Sauvegarder en JSON
                with open(os.path.join(self.save_results, "global_energy_consumption.json"), 'w') as f:
                    json.dump(total_energy_consumption, f, indent=4)
                
                # Créer et sauvegarder le rapport texte
                report = f"""=== Energy Consumption Metrics ===
                Total Duration: {duration:.2f} seconds
                Average CPU Power: {avg_power['cpu']:.2f} Watts
                Average RAM Power: {avg_power['ram']:.2f} Watts
                Average GPU Power: {avg_power['gpu']:.2f} Watts
                Total Energy Consumption: {total_energy_megajoules:.4f} Mega Joules

                === Measurement Details ===
                Start Time: {start_measurement['timestamp']}
                End Time: {end_measurement['timestamp']}
                Measurement Types:
                - CPU: {start_measurement['measurement_type']['cpu']}
                - RAM: {start_measurement['measurement_type']['ram']}
                - GPU: {start_measurement['measurement_type']['gpu']}
                """
                
                # Sauvegarder le rapport texte
                with open(os.path.join(self.save_results, "energy_report.txt"), 'w') as f:
                    f.write(report)
