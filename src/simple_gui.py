import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

import pygame
import numpy as np

from util import *


class Grid:

    def __init__(self):
        self.surface = pygame.Surface((GRID_SIZE, GRID_SIZE))
        self._grid = [[0 for i in range(28)] for j in range(28)]


    def draw(self, window):
        pygame.draw.rect(window, GRAY, self.surface.get_rect())
        line_coords = []

        for row_idx, row in enumerate(self._grid):
            for col_idx, value in enumerate(row):
                if value is 1:
                    pygame.draw.rect(window, RED, pygame.Rect(row_idx*CELL_SIZE, col_idx*CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for i in range(0, 29):
            pygame.draw.line(window, (0,0,0), [0, i * CELL_SIZE], [GRID_SIZE, i * CELL_SIZE])  # Horizontal
            pygame.draw.line(window, (0,0,0), [i * CELL_SIZE, 0], [i * CELL_SIZE, GRID_SIZE])  # Vertical


    def update(self, m_x, m_y):
        c_x, c_y = self.cell_index(m_x, m_y)
        keys_pressed = pygame.key.get_pressed()
        value = C_FULL if keys_pressed[pygame.K_r] else C_EMPTY if keys_pressed[pygame.K_t] else None
        if value is not None and self.get_cell(c_x, c_y) != value:
            logger.info("set cell (%d, %d) value to %d", c_x, c_y, value)
            self.set_cell(c_x, c_y, value)
            return True
        return False

    
    def clear_grid(self):
        self._grid = [[0 for i in range(28)] for j in range(28)]


    def cell_index(self, m_x, m_y):
        return int(m_x / CELL_SIZE), int(m_y / CELL_SIZE)


    def set_cell(self, c_x, c_y, value=0):
        self._grid[c_x][c_y] = value


    def get_cell(self, c_x, c_y):
        return self._grid[c_x][c_y]


    def get_grid(self):
        return self._grid


    def get_grid_np_array(self):
        arr = np.array(self._grid, dtype=np.uint8).transpose()
        return arr.reshape((1, IMG_SIZE))


    def from_np_array(self, np_arr):
        reshaped = np_arr.astype(int).reshape((NCELLS, NCELLS)).transpose()
        self._grid = reshaped.tolist()


class SimpleButton:

    def __init__(self, x, y, width, height, color, text = "Default", event=None):
        self.surface = pygame.Surface((width, height))
        self.rect = pygame.Rect(0, 0, width, height)
        self.x, self.y = x, y
        self.color = color
        self.text = text
        self._on_click_event = event


    def draw(self, sp_surface):
        pygame.draw.rect(self.surface, self.color, self.rect)

        mfont = pygame.font.Font(pygame.font.get_default_font(), 20)
        text_surface = mfont.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=((self.rect.width / 2), (self.rect.height / 2)))
        self.surface.blit(text_surface, text_rect)

        sp_surface.blit(self.surface, (self.x, self.y))


    def on_click(self):
        logger.info("Clicked %s button", self.text)
        if self._on_click_event:
            event = pygame.event.Event(pygame.USEREVENT + self._on_click_event)
            logger.debug("Posting event %s", event)
            pygame.event.post(event)
        else:
            logger.debug("On click event missing!")


class PredictionBox:

    def __init__(self):
        self.surface = pygame.Surface((SIDEPANEL_WIDTH, SIDEPANEL_WIDTH))
        self.rect = pygame.Rect(0, 0, SIDEPANEL_WIDTH, SIDEPANEL_WIDTH)
        self.prediction = 0


    def draw(self, surface):
        pygame.draw.rect(self.surface, LIGHT_GRAY, self.rect)

        # Title
        title_font = pygame.font.Font(pygame.font.get_default_font(), 40)
        text_surface = title_font.render("Prediction:", True, BLACK)
        self.surface.blit(text_surface, (0, 0))

        # Prediction
        pred_font = pygame.font.Font(pygame.font.get_default_font(), 100)
        text_surface = pred_font.render(str(self.prediction), True, BLACK)
        text_rect = text_surface.get_rect(center=((self.rect.width / 2), (self.rect.height / 2)))
        self.surface.blit(text_surface, text_rect)
        
        surface.blit(self.surface, (0, WINDOW_HEIGHT-SIDEPANEL_WIDTH))
        

class SidePanel:

    def __init__(self):
        self.surface = pygame.Surface((SIDEPANEL_WIDTH, WINDOW_HEIGHT))
        self.rect = pygame.Rect(0, 0, SIDEPANEL_WIDTH, WINDOW_HEIGHT)
        self.clear_button = SimpleButton(10, 10, SIDEPANEL_WIDTH - 20, 80, GREEN, "Clear", EVENT_CLEAR_GRID)
        self.example_button = SimpleButton(10, 110, SIDEPANEL_WIDTH - 20, 80, GREEN, "Example", EVENT_LOAD_EXAMPLE)
        self.buttons = [self.example_button, self.clear_button]
        self.pred_box = PredictionBox()


    def draw(self, window):
        pygame.draw.rect(self.surface, LIGHT_GRAY, self.rect)
        for button in self.buttons:
            button.draw(self.surface)
        self.pred_box.draw(self.surface)
        
        window.blit(self.surface, (SIDEPANEL_OFFSET, 0))


    def on_click(self, m_x, m_y):
        # TODO: MouseButton clicks
        new_x = m_x - SIDEPANEL_OFFSET
        for button in self.buttons:
            if button.surface.get_rect().collidepoint((m_x - SIDEPANEL_OFFSET - button.x, m_y - button.y)):
                button.on_click()


    def update(self):
        pass
    