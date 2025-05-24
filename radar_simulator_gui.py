import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from scipy.constants import c
from mpl_toolkits.mplot3d import Axes3D
import time

class RadarSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Simulator Pro")
        self.root.geometry("1200x800")
        
        # Параметры по умолчанию
        self.defaults = {
            'frequency': 5.0,  # ГГц
            'bandwidth': 50.0,  # МГц
            'pulse_width': 100.0,  # мкс
            'prf': 1000,  # Гц
            'rcs': 5.0,  # м²
            'velocity': 7500,  # м/с
            'distance': 1000,  # км
            'rotation': 0.5  # рад/с
        }
        
        # Инициализация модели РЛС
        self.radar = None
        self.results = None
        
        self.create_widgets()
        self.setup_plots()
    
    def create_widgets(self):
        """Создаем все элементы интерфейса"""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Левая панель - параметры
        param_frame = ttk.LabelFrame(main_frame, text="Параметры симуляции")
        param_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Правая панель - графики
        graph_frame = ttk.Frame(main_frame)
        graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем вкладки для разных типов параметров
        notebook = ttk.Notebook(param_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка параметров РЛС
        radar_tab = ttk.Frame(notebook)
        self.create_radar_controls(radar_tab)
        notebook.add(radar_tab, text="РЛС")
        
        # Вкладка параметров цели
        target_tab = ttk.Frame(notebook)
        self.create_target_controls(target_tab)
        notebook.add(target_tab, text="Цель")
        
        # Вкладка обработки
        processing_tab = ttk.Frame(notebook)
        self.create_processing_controls(processing_tab)
        notebook.add(processing_tab, text="Обработка")
        
        # Кнопки управления
        button_frame = ttk.Frame(param_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Запуск", command=self.run_simulation).pack(side=tk.LEFT, expand=True)
        ttk.Button(button_frame, text="Сброс", command=self.reset_parameters).pack(side=tk.LEFT, expand=True)
        ttk.Button(button_frame, text="Выход", command=self.root.quit).pack(side=tk.LEFT, expand=True)
        
        # Область графиков
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=graph_frame)
        self.graph_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Статус бар
        self.status_bar = ttk.Label(self.root, text="Готов к работе", relief=tk.SUNKEN)
        self.status_bar.pack(fill=tk.X)
    
    def create_radar_controls(self, parent):
        """Элементы управления параметрами РЛС"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.controls = {}
        
        # Частота
        ttk.Label(frame, text="Несущая частота (ГГц):").grid(row=0, column=0, sticky=tk.W)
        self.controls['frequency'] = ttk.Entry(frame)
        self.controls['frequency'].grid(row=0, column=1)
        self.controls['frequency'].insert(0, str(self.defaults['frequency']))
        
        # Полоса
        ttk.Label(frame, text="Полоса сигнала (МГц):").grid(row=1, column=0, sticky=tk.W)
        self.controls['bandwidth'] = ttk.Entry(frame)
        self.controls['bandwidth'].grid(row=1, column=1)
        self.controls['bandwidth'].insert(0, str(self.defaults['bandwidth']))
        
        # Длительность импульса
        ttk.Label(frame, text="Длительность импульса (мкс):").grid(row=2, column=0, sticky=tk.W)
        self.controls['pulse_width'] = ttk.Entry(frame)
        self.controls['pulse_width'].grid(row=2, column=1)
        self.controls['pulse_width'].insert(0, str(self.defaults['pulse_width']))
        
        # PRF
        ttk.Label(frame, text="Частота повторения (Гц):").grid(row=3, column=0, sticky=tk.W)
        self.controls['prf'] = ttk.Entry(frame)
        self.controls['prf'].grid(row=3, column=1)
        self.controls['prf'].insert(0, str(self.defaults['prf']))
    
    def create_target_controls(self, parent):
        """Элементы управления параметрами цели"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ЭПР
        ttk.Label(frame, text="ЭПР цели (м²):").grid(row=0, column=0, sticky=tk.W)
        self.controls['rcs'] = ttk.Entry(frame)
        self.controls['rcs'].grid(row=0, column=1)
        self.controls['rcs'].insert(0, str(self.defaults['rcs']))
        
        # Скорость
        ttk.Label(frame, text="Радиальная скорость (м/с):").grid(row=1, column=0, sticky=tk.W)
        self.controls['velocity'] = ttk.Entry(frame)
        self.controls['velocity'].grid(row=1, column=1)
        self.controls['velocity'].insert(0, str(self.defaults['velocity']))
        
        # Дистанция
        ttk.Label(frame, text="Дистанция до цели (км):").grid(row=2, column=0, sticky=tk.W)
        self.controls['distance'] = ttk.Entry(frame)
        self.controls['distance'].grid(row=2, column=1)
        self.controls['distance'].insert(0, str(self.defaults['distance']))
        
        # Вращение
        ttk.Label(frame, text="Скорость вращения (рад/с):").grid(row=3, column=0, sticky=tk.W)
        self.controls['rotation'] = ttk.Entry(frame)
        self.controls['rotation'].grid(row=3, column=1)
        self.controls['rotation'].insert(0, str(self.defaults['rotation']))
    
    def create_processing_controls(self, parent):
        """Элементы управления обработкой"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Размер БПФ
        ttk.Label(frame, text="Размер БПФ:").grid(row=0, column=0, sticky=tk.W)
        self.controls['fft_size'] = ttk.Combobox(frame, values=[256, 512, 1024, 2048])
        self.controls['fft_size'].grid(row=0, column=1)
        self.controls['fft_size'].current(2)  # 1024
        
        # Включение графиков
        self.controls['show_3d'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="Показывать 3D модель", 
                       variable=self.controls['show_3d']).grid(row=1, columnspan=2, sticky=tk.W)
    
    def setup_plots(self):
        """Настройка области графиков"""
        self.figure = plt.figure(figsize=(10, 8))
        self.figure.tight_layout(pad=3.0)
        
        # Создаем 4 области для графиков
        self.ax1 = self.figure.add_subplot(221)
        self.ax2 = self.figure.add_subplot(222)
        self.ax3 = self.figure.add_subplot(223)
        self.ax4 = self.figure.add_subplot(224, projection='3d')
        
        # Настройка подписей
        self.ax1.set_title("Зондирующий сигнал")
        self.ax2.set_title("Отраженный сигнал")
        self.ax3.set_title("Сжатый импульс")
        self.ax4.set_title("3D модель объекта")
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.grid(True)
    
    def get_parameters(self):
        """Получаем параметры из интерфейса"""
        params = {}
        try:
            # Параметры РЛС
            params['fc'] = float(self.controls['frequency'].get()) * 1e9
            params['bw'] = float(self.controls['bandwidth'].get()) * 1e6
            params['pulse_width'] = float(self.controls['pulse_width'].get()) * 1e-6
            params['prf'] = float(self.controls['prf'].get())
            
            # Параметры цели
            params['rcs'] = float(self.controls['rcs'].get())
            params['velocity'] = float(self.controls['velocity'].get())
            params['distance'] = float(self.controls['distance'].get()) * 1e3
            params['rotation'] = float(self.controls['rotation'].get())
            
            # Параметры обработки
            params['fft_size'] = int(self.controls['fft_size'].get())
            params['show_3d'] = self.controls['show_3d'].get()
            
            return params
        except ValueError as e:
            messagebox.showerror("Ошибка", f"Некорректные параметры: {str(e)}")
            return None
    
    def run_simulation(self):
        """Основная функция запуска симуляции"""
        params = self.get_parameters()
        if params is None:
            return
        
        self.status_bar.config(text="Идет моделирование...")
        self.root.update()
        
        try:
            # Инициализация РЛС с текущими параметрами
            self.radar = RadarSimulator(
                fc=params['fc'],
                bw=params['bw'],
                pulse_width=params['pulse_width'],
                prf=params['prf']
            )
            
            # Моделирование цели
            self.radar.simulate_target(
                rcs=params['rcs'],
                velocity=params['velocity'],
                rotation_rate=params['rotation']
            )
            
            # Генерация и обработка сигнала
            tx_signal = self.radar.generate_chirp()
            rx_signal = self.radar.propagate_signal(tx_signal, params['distance'])
            self.results = self.radar.process_signal(rx_signal)
            
            # Обновление графиков
            self.update_plots(tx_signal, rx_signal, params)
            
            self.status_bar.config(text="Моделирование завершено успешно")
        except Exception as e:
            self.status_bar.config(text="Ошибка моделирования")
            messagebox.showerror("Ошибка", str(e))
    
    def update_plots(self, tx_signal, rx_signal, params):
        """Обновление графиков"""
        # Очистка предыдущих графиков
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # График 1: Зондирующий сигнал
        t_tx = np.linspace(0, params['pulse_width'], len(tx_signal))
        self.ax1.plot(t_tx, np.real(tx_signal))
        self.ax1.set_title("Зондирующий сигнал")
        self.ax1.set_xlabel("Время (с)")
        self.ax1.set_ylabel("Амплитуда")
        self.ax1.grid(True)
        
        # График 2: Отраженный сигнал
        t_rx = np.linspace(0, params['pulse_width'], len(rx_signal))
        self.ax2.plot(t_rx, np.real(rx_signal))
        self.ax2.set_title("Отраженный сигнал")
        self.ax2.set_xlabel("Время (с)")
        self.ax2.grid(True)
        
        # График 3: Сжатый импульс
        self.ax3.plot(np.abs(self.results['compressed']))
        self.ax3.set_title("Сжатый импульс")
        self.ax3.set_xlabel("Отсчеты")
        self.ax3.set_ylabel("Амплитуда")
        self.ax3.grid(True)
        
        # График 4: 3D модель (если включено)
        if params['show_3d']:
            self.plot_3d_model()
        
        # Обновляем canvas
        self.figure.tight_layout()
        self.graph_canvas.draw()
    
    def plot_3d_model(self):
        """Генерация 3D модели объекта"""
        if self.results is None:
            return
            
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(0, 1, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = np.sqrt(self.results['rcs']) * np.cos(theta_grid)
        y_grid = np.sqrt(self.results['rcs']) * np.sin(theta_grid)
        
        self.ax4.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)
        self.ax4.set_title("3D модель объекта")
        self.ax4.set_xlabel("X")
        self.ax4.set_ylabel("Y")
        self.ax4.set_zlabel("Z")
    
    def reset_parameters(self):
        """Сброс параметров к значениям по умолчанию"""
        for key, entry in self.controls.items():
            if isinstance(entry, ttk.Entry):
                entry.delete(0, tk.END)
                entry.insert(0, str(self.defaults.get(key, 0)))
        
        self.status_bar.config(text="Параметры сброшены к значениям по умолчанию")

class RadarSimulator:
    def __init__(self, fc=5e9, bw=50e6, pulse_width=100e-6, prf=1000):
        """Инициализация параметров РЛС"""
        self.fc = fc  # Несущая частота (Гц)
        self.bw = bw  # Полоса (Гц)
        self.pulse_width = pulse_width  # Длительность импульса (с)
        self.prf = prf  # Частота повторения (Гц)
        self.wavelength = c / fc  # Длина волны
        self.range_res = c / (2 * bw)  # Разрешение по дальности
        
    def generate_chirp(self):
        """Генерация ЛЧМ сигнала"""
        t = np.linspace(0, self.pulse_width, int(self.pulse_width * 1e6))
        return signal.chirp(t, f0=self.fc-self.bw/2, f1=self.fc+self.bw/2, 
                          t1=self.pulse_width, method='linear')

    def simulate_target(self, rcs=1.0, velocity=0, rotation_rate=0):
        """Модель космического объекта"""
        self.rcs = rcs  # ЭПР (м²)
        self.velocity = velocity  # Радиальная скорость (м/с)
        self.rotation_rate = rotation_rate  # Скорость вращения (рад/с)
        
    def propagate_signal(self, tx_signal, distance):
        """Модель распространения сигнала"""
        delay = 2 * distance / c
        doppler_shift = 2 * self.velocity / self.wavelength
        
        # Добавление доплеровского смещения
        t = np.linspace(0, len(tx_signal)/1e6, len(tx_signal))
        rx_signal = np.sqrt(self.rcs) * np.roll(tx_signal, int(delay*1e6)) * \
                   np.exp(1j*2*np.pi*doppler_shift*t)
        
        # Добавление микродоплеровских эффектов
        if self.rotation_rate > 0:
            microdoppler = 0.1 * np.sin(2*np.pi*self.rotation_rate*t)
            rx_signal *= np.exp(1j*2*np.pi*microdoppler*t)
            
        # Добавление шумов
        noise_power = 0.1
        rx_signal += np.sqrt(noise_power/2) * \
                    (np.random.randn(len(tx_signal)) + \
                    1j*np.random.randn(len(tx_signal)))
        
        return rx_signal

    def process_signal(self, rx_signal):
        """Обработка сигнала и извлечение НКИ"""
        # Сжатие импульса
        compressed = np.correlate(rx_signal, self.generate_chirp(), mode='same')
        
        # Доплеровская обработка
        fft_size = 1024
        doppler_spectrum = np.fft.fftshift(np.abs(np.fft.fft(compressed, fft_size)))
        doppler_bins = np.fft.fftshift(np.fft.fftfreq(fft_size, 1/1e6))
        
        # Оценка параметров
        estimated_velocity = self._estimate_velocity(doppler_spectrum, doppler_bins)
        estimated_rcs = self._estimate_rcs(compressed)
        
        return {
            'velocity': estimated_velocity,
            'rcs': estimated_rcs,
            'spectrum': doppler_spectrum,
            'compressed': compressed
        }
    
    def _estimate_velocity(self, spectrum, bins):
        """Оценка скорости по доплеровскому спектру"""
        peak_idx = np.argmax(spectrum)
        return bins[peak_idx] * self.wavelength / 2
        
    def _estimate_rcs(self, compressed_signal):
        """Оценка ЭПР по мощности сигнала"""
        peak_power = np.max(np.abs(compressed_signal)**2)
        return peak_power  # Упрощенная модель

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarSimulatorGUI(root)
    root.mainloop()