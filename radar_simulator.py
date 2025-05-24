import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.constants import c
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import time
import pandas as pd

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
            microdoppler = 0.01 * np.sin(2*np.pi*self.rotation_rate*t)
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

    def visualize_results(self, results):
        """Визуализация результатов"""
        plt.figure(figsize=(15, 10))
        
        # График сжатого импульса
        plt.subplot(2, 2, 1)
        plt.plot(np.abs(results['compressed']))
        plt.title("Сжатый импульс")
        plt.xlabel("Отсчеты")
        plt.ylabel("Амплитуда")
        
        # Доплеровский спектр
        plt.subplot(2, 2, 2)
        plt.plot(results['spectrum'])
        plt.title("Доплеровский спектр")
        plt.xlabel("Доплеровская частота")
        plt.ylabel("Мощность")
        
        # 3D модель объекта
        plt.subplot(2, 2, 3, projection='3d')
        self._plot_3d_model()
        
        # Поляризационная матрица
        plt.subplot(2, 2, 4)
        self._plot_polarization_matrix()
        
        plt.tight_layout()
        plt.show()
        
    def _plot_3d_model(self):
        """Генерация 3D модели объекта"""
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(0, 1, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = np.sqrt(self.rcs) * np.cos(theta_grid)
        y_grid = np.sqrt(self.rcs) * np.sin(theta_grid)
        
        ax = plt.gca()
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5)
        ax.set_title("3D модель объекта")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
    def _plot_polarization_matrix(self):
        """Визуализация поляризационной матрицы"""
        matrix = np.array([[1.0, 0.2], [0.2, 0.5]])  # Пример матрицы
        plt.imshow(matrix, cmap='viridis')
        plt.colorbar()
        plt.title("Поляризационная матрица рассеяния")
        plt.xticks([0, 1], ['HH', 'HV'])
        plt.yticks([0, 1], ['VH', 'VV'])

    def interactive_plot(self, results):
        """Интерактивная визуализация с Plotly"""
        fig = go.Figure()
        
        # Сжатый импульс
        fig.add_trace(go.Scatter(
            y=np.abs(results['compressed']),
            name='Сжатый импульс',
            xaxis='x1',
            yaxis='y1'
        ))
        
        # Доплеровский спектр
        fig.add_trace(go.Scatter(
            y=results['spectrum'],
            name='Доплеровский спектр',
            xaxis='x2',
            yaxis='y2'
        ))
        
        # 3D модель
        theta = np.linspace(0, 2*np.pi, 30)
        z = np.linspace(0, 1, 30)
        theta_grid, z_grid = np.meshgrid(theta, z)
        x_grid = np.sqrt(self.rcs) * np.cos(theta_grid)
        y_grid = np.sqrt(self.rcs) * np.sin(theta_grid)
        
        fig.add_trace(go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            name='3D модель',
            colorscale='Viridis',
            showscale=False,
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Результаты обработки РЛС',
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )
        
        fig.show()

# Пример использования
if __name__ == "__main__":
    # Инициализация РЛС
    radar = RadarSimulator(fc=5e9, bw=50e6, pulse_width=100e-6)
    
    # Моделирование цели
    radar.simulate_target(rcs=5.0, velocity=7500, rotation_rate=0.5)
    
    # Генерация и обработка сигнала
    tx_signal = radar.generate_chirp()
    rx_signal = radar.propagate_signal(tx_signal, distance=1000e3)
    results = radar.process_signal(rx_signal)
    
    # Вывод результатов
    print(f"Оцененная скорость: {results['velocity']:.2f} м/с")
    print(f"Оцененная ЭПР: {results['rcs']:.2f} м²")
    
    # Визуализация
    radar.visualize_results(results)
    
    # Интерактивный график (требуется plotly)
    # radar.interactive_plot(results)