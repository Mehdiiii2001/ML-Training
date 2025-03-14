import pygame
import os
from gtts import gTTS
from googletrans import Translator
import math
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import time
import arabic_reshaper
from bidi.algorithm import get_display
import requests
from io import BytesIO
import json
import numpy as np
from pygame import gfxdraw
import colorsys
import cv2

class ParticleSystem:
    def __init__(self, x, y, color, particle_type="normal"):
        self.particles = []
        self.origin_x = x
        self.origin_y = y
        self.color = color
        self.particle_type = particle_type
        self.spawn_radius = 50
        
    def add_particle(self):
        if self.particle_type == "normal":
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, self.spawn_radius)
            x = self.origin_x + math.cos(angle) * distance
            y = self.origin_y + math.sin(angle) * distance
            self.particles.append(Particle(x, y, self.color, self.particle_type))
        elif self.particle_type == "neural":
            self.particles.append(NeuralParticle(self.origin_x, self.origin_y, self.color))
        elif self.particle_type == "data":
            self.particles.append(DataParticle(self.origin_x, self.origin_y, self.color))

    def update(self, screen):
        for particle in self.particles[:]:
            particle.move()
            particle.draw(screen)
            if particle.is_dead():
                self.particles.remove(particle)

class Particle:
    def __init__(self, x, y, color, particle_type="normal"):
        self.x = x
        self.y = y
        self.color = color
        self.size = random.randint(2, 6)
        self.speed = random.uniform(1, 4)
        self.angle = random.uniform(0, 2 * math.pi)
        self.life = 255
        self.decay_rate = random.uniform(2, 5)
        self.particle_type = particle_type

    def move(self):
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed
        self.life -= self.decay_rate
        
    def draw(self, screen):
        if not self.is_dead():
            alpha = max(0, min(255, int(self.life)))
            particle_color = (*self.color, alpha)
            gfxdraw.filled_circle(screen, int(self.x), int(self.y), int(self.size), particle_color)
            
    def is_dead(self):
        return self.life <= 0

class NeuralParticle(Particle):
    def __init__(self, x, y, color):
        super().__init__(x, y, color, "neural")
        self.connections = []
        self.target_x = x + random.randint(-100, 100)
        self.target_y = y + random.randint(-100, 100)
        
    def move(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            self.x += (dx/distance) * self.speed
            self.y += (dy/distance) * self.speed
            
        self.life -= self.decay_rate
        
    def draw(self, screen):
        if not self.is_dead():
            alpha = max(0, min(255, int(self.life)))
            particle_color = (*self.color, alpha)
            gfxdraw.filled_circle(screen, int(self.x), int(self.y), int(self.size), particle_color)
            
            # Draw neural connections
            for other in self.connections:
                if not other.is_dead():
                    line_alpha = min(self.life, other.life)
                    line_color = (*self.color, int(line_alpha * 0.5))
                    gfxdraw.line(screen, int(self.x), int(self.y), 
                               int(other.x), int(other.y), line_color)

class DataParticle(Particle):
    def __init__(self, x, y, color):
        super().__init__(x, y, color, "data")
        self.binary_data = "".join(random.choice("01") for _ in range(8))
        self.font = pygame.font.Font(None, 20)
        
    def draw(self, screen):
        if not self.is_dead():
            alpha = max(0, min(255, int(self.life)))
            text_surface = self.font.render(self.binary_data, True, (*self.color, alpha))
            screen.blit(text_surface, (self.x, self.y))

class AIBackground:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particle_systems = []
        self.neural_connections = []
        self.time = 0
        self.setup_particle_systems()
        
    def setup_particle_systems(self):
        # Create neural network nodes
        for _ in range(5):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            color = self.generate_neon_color()
            system = ParticleSystem(x, y, color, "neural")
            self.particle_systems.append(system)
            
        # Create data streams
        for _ in range(3):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            color = self.generate_neon_color()
            system = ParticleSystem(x, y, color, "data")
            self.particle_systems.append(system)
    
    def generate_neon_color(self):
        hue = random.random()
        saturation = 1.0
        value = 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return tuple(int(x * 255) for x in rgb)
    
    def update(self, screen):
        self.time += 0.01
        
        # Update existing particle systems
        for system in self.particle_systems:
            if random.random() < 0.1:
                system.add_particle()
            system.update(screen)
            
        # Randomly create new neural connections
        if random.random() < 0.05:
            self.create_neural_connection()
            
    def create_neural_connection(self):
        neural_systems = [s for s in self.particle_systems if s.particle_type == "neural"]
        if len(neural_systems) >= 2:
            source = random.choice(neural_systems)
            target = random.choice(neural_systems)
            if source != target:
                for p1 in source.particles:
                    for p2 in target.particles:
                        if random.random() < 0.1:
                            p1.connections.append(p2)

class AIAnimatedBackground:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.patterns = []
        self.time = 0
        self.matrix_chars = "01"
        self.setup_patterns()
        
    def setup_patterns(self):
        # Matrix rain effect
        self.matrix_streams = []
        for x in range(0, self.width, 20):
            speed = random.uniform(5, 15)
            self.matrix_streams.append({
                'x': x,
                'y': random.randint(-100, 0),
                'speed': speed,
                'chars': [random.choice(self.matrix_chars) for _ in range(20)]
            })
            
        # Neural network visualization
        self.nodes = []
        self.connections = []
        for _ in range(20):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            self.nodes.append({
                'x': x,
                'y': y,
                'size': random.randint(3, 8),
                'color': self.generate_neon_color()
            })
        
        # Create random connections between nodes
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if random.random() < 0.2:
                    self.connections.append((i, j))
    
    def generate_neon_color(self):
        hue = random.random()
        saturation = 1.0
        value = 1.0
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        return tuple(int(x * 255) for x in rgb)
    
    def draw_matrix_rain(self, screen):
        font = pygame.font.Font(None, 24)
        for stream in self.matrix_streams:
            x, y = stream['x'], stream['y']
            for i, char in enumerate(stream['chars']):
                alpha = max(0, min(255, 255 - i * 10))
                color = (0, 255, 100, alpha)
                text = font.render(char, True, color)
                pos_y = y + i * 20
                if 0 <= pos_y <= self.height:
                    text.set_alpha(alpha)
                    screen.blit(text, (x, pos_y))
            
            stream['y'] += stream['speed']
            if stream['y'] > self.height + 100:
                stream['y'] = random.randint(-100, 0)
                stream['chars'] = [random.choice(self.matrix_chars) for _ in range(20)]
    
    def draw_neural_network(self, screen):
        # Draw connections
        for conn in self.connections:
            start = self.nodes[conn[0]]
            end = self.nodes[conn[1]]
            
            # Calculate pulse position
            pulse_pos = (math.sin(self.time + conn[0] * 0.1) + 1) / 2
            
            # Draw connection line with pulse
            start_pos = (int(start['x']), int(start['y']))
            end_pos = (int(end['x']), int(end['y']))
            
            # Calculate pulse position along the line
            pulse_x = start['x'] + (end['x'] - start['x']) * pulse_pos
            pulse_y = start['y'] + (end['y'] - start['y']) * pulse_pos
            
            # Draw base line
            pygame.draw.line(screen, (30, 30, 80), start_pos, end_pos, 1)
            
            # Draw pulse
            pulse_radius = 3
            pulse_color = (*start['color'][:3], 150)
            pygame.draw.circle(screen, pulse_color, (int(pulse_x), int(pulse_y)), pulse_radius)
        
        # Draw nodes
        for node in self.nodes:
            glow_radius = node['size'] * 2
            base_color = node['color']
            
            # Draw glow
            for r in range(glow_radius, 0, -1):
                alpha = int(100 * (1 - r/glow_radius))
                color = (*base_color[:3], alpha)
                pygame.draw.circle(screen, color, (int(node['x']), int(node['y'])), r)
            
            # Draw node core
            pygame.draw.circle(screen, base_color, (int(node['x']), int(node['y'])), node['size'])
    
    def draw_circuit_pattern(self, screen):
        # Draw circuit-like patterns
        for i in range(0, self.width, 50):
            for j in range(0, self.height, 50):
                if random.random() < 0.3:
                    color = (0, 150, 255, 50)
                    points = [
                        (i, j),
                        (i + 50, j),
                        (i + 50, j + 50),
                        (i, j + 50)
                    ]
                    pygame.draw.lines(screen, color, True, points, 1)
    
    def update(self, screen):
        self.time += 0.02
        
        # Create a new surface for the background
        background = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        
        # Draw circuit patterns
        self.draw_circuit_pattern(background)
        
        # Draw matrix rain
        self.draw_matrix_rain(background)
        
        # Draw neural network
        self.draw_neural_network(background)
        
        # Apply the background to the screen with alpha blending
        screen.blit(background, (0, 0))

class AISceneGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.scenes = [
            self.create_neural_scene,
            self.create_data_flow_scene,
            self.create_quantum_scene,
            self.create_cybernetic_scene,
            self.create_matrix_scene,
            self.create_deep_learning_scene
        ]
        self.current_scene = random.choice(self.scenes)
        self.scene_duration = 10  # seconds
        self.last_scene_change = time.time()
        self.transition_alpha = 255
        self.next_scene = None
        
    def create_neural_scene(self, screen):
        # Advanced neural network visualization
        nodes = []
        connections = []
        layers = 4
        nodes_per_layer = 5
        
        # Create nodes in layers
        for layer in range(layers):
            layer_nodes = []
            x_base = (layer + 1) * self.width // (layers + 1)
            for i in range(nodes_per_layer):
                y = (i + 1) * self.height // (nodes_per_layer + 1)
                color = self.generate_ai_color()
                nodes.append({
                    'x': x_base + random.randint(-20, 20),
                    'y': y + random.randint(-20, 20),
                    'color': color,
                    'size': random.randint(4, 8),
                    'layer': layer
                })
                layer_nodes.append(len(nodes) - 1)
            
            # Connect to previous layer
            if layer > 0:
                for curr in layer_nodes:
                    for prev in range(len(nodes) - nodes_per_layer - 1, len(nodes) - nodes_per_layer * 2, -1):
                        if random.random() < 0.6:
                            connections.append((prev, curr))
        
        # Draw connections with data flow
        for conn in connections:
            start = nodes[conn[0]]
            end = nodes[conn[1]]
            
            # Animated data pulse
            pulse_pos = (math.sin(time.time() * 2 + conn[0]) + 1) / 2
            pulse_x = start['x'] + (end['x'] - start['x']) * pulse_pos
            pulse_y = start['y'] + (end['y'] - start['y']) * pulse_pos
            
            # Draw connection
            pygame.draw.line(screen, (30, 40, 100), 
                           (start['x'], start['y']), 
                           (end['x'], end['y']), 2)
            
            # Draw pulse
            pygame.draw.circle(screen, (100, 200, 255), 
                             (int(pulse_x), int(pulse_y)), 4)
        
        # Draw nodes with glow
        for node in nodes:
            self.draw_glowing_node(screen, node)
    
    def create_data_flow_scene(self, screen):
        # Data streams visualization
        streams = []
        num_streams = 15
        
        for _ in range(num_streams):
            stream = {
                'start': (random.randint(0, self.width), random.randint(0, self.height)),
                'length': random.randint(100, 300),
                'angle': random.uniform(0, 2 * math.pi),
                'speed': random.uniform(2, 5),
                'color': self.generate_ai_color(),
                'data': "".join(random.choice("01") for _ in range(8))
            }
            streams.append(stream)
        
        # Draw data streams
        font = pygame.font.Font(None, 20)
        for stream in streams:
            x, y = stream['start']
            angle = stream['angle']
            
            # Draw flowing data
            for i in range(len(stream['data'])):
                offset = (time.time() * stream['speed'] + i * 20) % stream['length']
                pos_x = x + math.cos(angle) * offset
                pos_y = y + math.sin(angle) * offset
                
                if 0 <= pos_x <= self.width and 0 <= pos_y <= self.height:
                    char = stream['data'][i]
                    text = font.render(char, True, stream['color'])
                    screen.blit(text, (pos_x, pos_y))
    
    def create_quantum_scene(self, screen):
        # Quantum computing visualization
        qubits = []
        num_qubits = 8
        
        # Create quantum states
        for i in range(num_qubits):
            angle = i * (2 * math.pi / num_qubits)
            radius = min(self.width, self.height) * 0.3
            x = self.width/2 + math.cos(angle) * radius
            y = self.height/2 + math.sin(angle) * radius
            qubits.append({
                'pos': (x, y),
                'state': random.random(),
                'color': self.generate_ai_color()
            })
        
        # Draw quantum entanglement
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if random.random() < 0.3:
                    start = qubits[i]['pos']
                    end = qubits[j]['pos']
                    
                    # Quantum entanglement effect
                    points = []
                    steps = 20
                    for step in range(steps + 1):
                        t = step / steps
                        wave = math.sin(t * math.pi * 2 + time.time() * 3) * 20
                        x = start[0] + (end[0] - start[0]) * t
                        y = start[1] + (end[1] - start[1]) * t + wave
                        points.append((x, y))
                    
                    if len(points) > 1:
                        pygame.draw.lines(screen, (100, 100, 255, 100), False, points, 2)
        
        # Draw qubits
        for qubit in qubits:
            x, y = qubit['pos']
            color = qubit['color']
            state = qubit['state']
            
            # Quantum state visualization
            radius = 10 + math.sin(time.time() * 2) * 3
            self.draw_quantum_state(screen, (x, y), radius, color, state)
    
    def create_cybernetic_scene(self, screen):
        # Cybernetic circuit visualization
        grid_size = 40
        for x in range(0, self.width, grid_size):
            for y in range(0, self.height, grid_size):
                if random.random() < 0.3:
                    color = self.generate_ai_color()
                    self.draw_circuit_element(screen, x, y, grid_size, color)
    
    def create_matrix_scene(self, screen):
        # Enhanced Matrix-style rain
        chars = "01アイウエオカキクケコサシスセソタチツテト"
        font = pygame.font.Font(None, 24)
        
        for x in range(0, self.width, 20):
            y = (time.time() * 100 + x) % (self.height + 200) - 100
            
            for i in range(15):
                char = random.choice(chars)
                alpha = max(0, min(255, 255 - i * 15))
                color = (0, 255, 100, alpha)
                
                text = font.render(char, True, color)
                text.set_alpha(alpha)
                screen.blit(text, (x, y + i * 20))
    
    def create_deep_learning_scene(self, screen):
        # Deep learning architecture visualization
        layers = [784, 512, 256, 128, 64, 10]  # Example neural network architecture
        max_neurons = max(layers)
        layer_spacing = self.width / (len(layers) + 1)
        
        # Draw connections between layers
        for i in range(len(layers) - 1):
            start_neurons = layers[i]
            end_neurons = layers[i + 1]
            
            start_x = (i + 1) * layer_spacing
            end_x = (i + 2) * layer_spacing
            
            for s in range(0, start_neurons, 10):  # Skip some neurons for clarity
                start_y = (s / start_neurons) * self.height
                
                for e in range(0, end_neurons, 10):
                    end_y = (e / end_neurons) * self.height
                    
                    # Animated connection strength
                    strength = (math.sin(time.time() * 2 + s + e) + 1) / 2
                    color = self.interpolate_color((30, 30, 80), (100, 100, 255), strength)
                    
                    pygame.draw.line(screen, color, 
                                   (start_x, start_y), 
                                   (end_x, end_y), 1)
    
    def generate_ai_color(self):
        # Generate colors suitable for AI visualization
        palettes = [
            # Cyber blue
            (0, 150, 255),
            # Neural green
            (0, 255, 150),
            # Quantum purple
            (150, 0, 255),
            # Data orange
            (255, 150, 0),
            # AI pink
            (255, 0, 150)
        ]
        return random.choice(palettes)
    
    def draw_glowing_node(self, screen, node):
        x, y = node['x'], node['y']
        color = node['color']
        size = node['size']
        
        # Draw glow
        max_radius = size * 3
        for r in range(max_radius, size - 1, -1):
            alpha = int(100 * (1 - r/max_radius))
            glow_color = (*color[:3], alpha)
            pygame.draw.circle(screen, glow_color, (int(x), int(y)), r)
        
        # Draw core
        pygame.draw.circle(screen, color, (int(x), int(y)), size)
    
    def draw_quantum_state(self, screen, pos, radius, color, state):
        x, y = pos
        
        # Draw quantum superposition effect
        num_circles = 3
        for i in range(num_circles):
            phase = time.time() * 2 + i * 2 * math.pi / num_circles
            offset_x = math.cos(phase) * radius * 0.3
            offset_y = math.sin(phase) * radius * 0.3
            
            alpha = int(255 * (1 - i/num_circles))
            circle_color = (*color[:3], alpha)
            
            pygame.draw.circle(screen, circle_color,
                             (int(x + offset_x), int(y + offset_y)),
                             int(radius * (1 - i/num_circles)))
    
    def draw_circuit_element(self, screen, x, y, size, color):
        # Draw various circuit elements
        element_type = random.choice(['node', 'gate', 'connection'])
        
        if element_type == 'node':
            pygame.draw.circle(screen, color, (x + size//2, y + size//2), size//4)
        elif element_type == 'gate':
            rect = pygame.Rect(x + size//4, y + size//4, size//2, size//2)
            pygame.draw.rect(screen, color, rect, 2)
        else:
            points = [
                (x, y),
                (x + size, y),
                (x + size, y + size),
                (x, y + size)
            ]
            pygame.draw.lines(screen, color, True, points, 2)
    
    def interpolate_color(self, color1, color2, factor):
        return tuple(int(c1 + (c2 - c1) * factor) for c1, c2 in zip(color1, color2))
    
    def update(self, screen):
        # Check if it's time to change scene
        current_time = time.time()
        if current_time - self.last_scene_change > self.scene_duration:
            if self.next_scene is None:
                self.next_scene = random.choice([s for s in self.scenes if s != self.current_scene])
                self.transition_alpha = 255
            
            # Handle scene transition
            self.transition_alpha = max(0, self.transition_alpha - 5)
            
            if self.transition_alpha == 0:
                self.current_scene = self.next_scene
                self.next_scene = None
                self.last_scene_change = current_time
                self.transition_alpha = 255
        
        # Create a new surface for the current scene
        scene_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.current_scene(scene_surface)
        
        # If transitioning, draw both scenes
        if self.next_scene is not None:
            next_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            self.next_scene(next_surface)
            
            # Blend between scenes
            alpha = 255 - self.transition_alpha
            next_surface.set_alpha(alpha)
            scene_surface.set_alpha(self.transition_alpha)
            
            screen.blit(next_surface, (0, 0))
        
        screen.blit(scene_surface, (0, 0))

def create_gradient_background(width, height, colors):
    """Create a dynamic gradient background with multiple colors"""
    background = pygame.Surface((width, height))
    
    num_colors = len(colors)
    segment_height = height / (num_colors - 1)
    
    for i in range(num_colors - 1):
        start_color = colors[i]
        end_color = colors[i + 1]
        start_y = int(i * segment_height)
        end_y = int((i + 1) * segment_height)
        
        for y in range(start_y, end_y):
            progress = (y - start_y) / (end_y - start_y)
            current_color = [
                int(start_color[j] + (end_color[j] - start_color[j]) * progress)
                for j in range(3)
            ]
            pygame.draw.line(background, current_color, (0, y), (width, y))
    
    return background

def apply_bloom_effect(surface, intensity=1.5):
    """Apply a bloom effect to make bright areas glow"""
    # Convert pygame surface to PIL Image
    string_image = pygame.image.tostring(surface, 'RGB')
    temp_surface = Image.frombytes('RGB', surface.get_size(), string_image)
    
    # Apply blur
    blurred = temp_surface.filter(ImageFilter.GaussianBlur(radius=5))
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(blurred)
    brightened = enhancer.enhance(intensity)
    
    # Convert back to pygame surface
    string_image = brightened.tobytes()
    return pygame.image.fromstring(string_image, surface.get_size(), 'RGB')

def render_persian_text(text, font, color):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return font.render(bidi_text, True, color)

def create_video_from_text(text, duration=15):
    try:
        # Initialize
        pygame.init()
        pygame.mixer.init()
        
        # Set up display
        width, height = 1280, 720
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Advanced AI Presentation")
        
        # Create translator instance
        translator = Translator()
        translation = translator.translate(text, dest='fa')
        translated_text = translation.text
        
        # Initialize backgrounds
        ai_scene = AISceneGenerator(width, height)
        
        # Load fonts
        try:
            font_path = "Vazir.ttf"
            if not os.path.exists(font_path):
                font_path = "arial.ttf"
            font = pygame.font.Font(font_path, 48)
            font_fa = pygame.font.Font(font_path, 54)
        except:
            font = pygame.font.Font(None, 48)
            font_fa = pygame.font.Font(None, 54)
        
        # Create text surfaces
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(width//2, height//3))
        
        translated_surface = render_persian_text(translated_text, font_fa, (255, 255, 0))
        translated_rect = translated_surface.get_rect(center=(width//2, 2*height//3))
        
        # Convert text to speech
        audio_file = "temp_audio.mp3"
        if os.path.exists(audio_file):
            try:
                os.remove(audio_file)
            except:
                audio_file = f"temp_audio_{random.randint(1000, 9999)}.mp3"
        
        tts = gTTS(text=text, lang='en')
        tts.save(audio_file)
        
        # Load and play audio
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
        except:
            print("Warning: Could not play audio")
        
        # Animation variables
        start_time = time.time()
        text_alpha = 0
        wave_offset = 0
        
        # Main loop
        clock = pygame.time.Clock()
        running = True
        
        while running:
            current_time = time.time() - start_time
            
            if current_time >= duration:
                running = False
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Clear screen
            screen.fill((0, 0, 20))
            
            # Update AI scene
            ai_scene.update(screen)
            
            # Apply bloom effect
            screen_with_bloom = apply_bloom_effect(screen)
            screen.blit(screen_with_bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
            
            # Fade in text
            text_alpha = min(255, text_alpha + 3)
            
            # Create temporary surface for text effects
            temp_surface = pygame.Surface((width, height), pygame.SRCALPHA)
            temp_surface.fill((0, 0, 0, 0))
            
            # Draw English text with enhanced wave effect
            wave_offset += 0.1
            for i in range(len(text)):
                char = text[i]
                char_surface = font.render(char, True, (255, 255, 255))
                y_offset = math.sin(wave_offset + i * 0.3) * 10
                x_offset = math.cos(wave_offset + i * 0.2) * 5
                x_pos = text_rect.x + i * 30 + x_offset
                y_pos = text_rect.y + y_offset
                temp_surface.blit(char_surface, (x_pos, y_pos))
            
            # Draw Persian text with subtle wave effect
            reshaped_text = arabic_reshaper.reshape(translated_text)
            bidi_text = get_display(reshaped_text)
            total_width = len(bidi_text) * 30
            start_x = (width - total_width) // 2
            
            for i, char in enumerate(bidi_text):
                char_surface = font_fa.render(char, True, (255, 255, 0))
                y_offset = math.sin(wave_offset + i * 0.2) * 5
                x_pos = start_x + i * 30
                y_pos = translated_rect.y + y_offset
                temp_surface.blit(char_surface, (x_pos, y_pos))
            
            # Apply alpha to temporary surface
            temp_surface.set_alpha(text_alpha)
            screen.blit(temp_surface, (0, 0))
            
            # Draw enhanced glowing effect around both texts
            for rect, color in [(text_rect, (100, 100, 255)), (translated_rect, (255, 255, 100))]:
                glow_radius = 25
                for r in range(glow_radius, 0, -2):
                    alpha = int(25 * (1 - r/glow_radius))
                    glow_color = (*color, alpha)
                    glow_surface = pygame.Surface((width, height), pygame.SRCALPHA)
                    pygame.draw.rect(glow_surface, glow_color,
                                   rect.inflate(r*2, r*2), border_radius=r)
                    screen.blit(glow_surface, (0, 0))
            
            pygame.display.flip()
            clock.tick(60)
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Cleanup
        try:
            pygame.mixer.music.stop()
        except:
            pass
        
        try:
            pygame.mixer.quit()
        except:
            pass
        
        try:
            pygame.quit()
        except:
            pass
        
        try:
            if os.path.exists(audio_file):
                os.remove(audio_file)
        except:
            pass

if __name__ == "__main__":
    print("=== Advanced AI Text-to-Video Presentation System ===")
    print("This system creates a professional visualization of text with:")
    print("- Dynamic AI-themed backgrounds")
    print("- Neural network visualizations")
    print("- Persian translation")
    print("- Professional text effects")
    print("- High-quality audio narration")
    print("================================================")
    
    while True:
        try:
            user_text = input("Enter the text you want to convert to video (or 'quit' to exit): ")
            if user_text.lower() == 'quit':
                break
            
            create_video_from_text(user_text, duration=20)
            
            # Wait a moment before starting the next iteration
            time.sleep(1)
            
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try again.")
            time.sleep(1)
            continue 