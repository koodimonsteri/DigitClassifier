import pickle
import random
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import pygame
import numpy as np

import data_decoder as dc
from simple_gui import SidePanel, Grid
from util import *
from nn import MLPModel, CNNModel

"""
-Game
--Grid
---2D list
--Sidepanel
---Clear, Example Buttons
---Prediction results
"""

class Game:

    def __init__(self):
        self.running = False
        self._grid = Grid()
        self._sidepanel = SidePanel()
        self._drawing = False
        self.model = CNNModel()
        self.example_images = []
        self._init_example_images(N_EXAMPLES)


    def _init_example_images(self, n_images):
        train_images = dc.get_images(DATA_DIR + TEST_IMAGES)
        train_images1 = train_images.reshape((len(train_images), NCELLS, NCELLS))
        train_images2 = train_images1 / 255.0
        train_images2 = np.where(train_images2 > 0.5, 1.0, 0.0)
        self.example_images = train_images2[:n_images]


    def _process_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False

        # Keyboard events
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.key == pygame.K_1:
                # MLP
                self.model = MLPModel()
                logger.info("Switched to MLP model")
            elif event.key == pygame.K_2:
                # CNN
                self.model = CNNModel()
                logger.info("Switched to CNN model")

        # Mouse events
        if event.type == pygame.MOUSEBUTTONDOWN:
            m_x, m_y = event.pos[0], event.pos[1]
            if self._sidepanel.surface.get_rect().collidepoint((m_x - SIDEPANEL_OFFSET, m_y)):
                self._sidepanel.on_click(m_x, m_y)

        # Pygame userevents
        if event.type == EVENT_CLEAR_GRID + pygame.USEREVENT:
            logger.info("Clearing grid!")
            self._grid.clear_grid()
        elif event.type == EVENT_LOAD_EXAMPLE + pygame.USEREVENT:
            logger.info("Loading example image!")
            img = random.choice(self.example_images)
            pred = self.model.predict(img)
            self._sidepanel.pred_box.prediction = pred
            self._grid.from_np_array(img)


    def _update(self):
        m_x, m_y = pygame.mouse.get_pos()
        # Update grid
        updated = False
        if self._grid.surface.get_rect().collidepoint((int(m_x), int(m_y))):
            updated = self._grid.update(m_x, m_y)
        # Predict if updated
        if updated:
            transposed = np.array(self._grid.get_grid(), dtype=np.uint8).transpose()
            pred = self.model.predict(transposed)
            self._sidepanel.pred_box.prediction = pred


    def _show(self, window):
        window.fill((50, 50, 50))

        self._grid.draw(window)
        self._sidepanel.draw(window)

        pygame.display.flip()


    def run(self):
        pygame.init()
        window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        clock = pygame.time.Clock()
        fps = 0
        dt = 0.0
        self.running = True
        while self.running:
            time_delta = clock.tick(60) / 1000.0
            dt += time_delta

            for event in pygame.event.get():
                self._process_event(event)

            self._update()

            self._show(window)

            fps += 1
            if dt >= 1.0:
                dt -= 1.0
                logger.info("-----FPS %d-----", fps)
                fps = 0
        
        pygame.quit()
        