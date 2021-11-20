#!/usr/bin/env ipython3
import os
import pygame

class Background():
    def __init__(self, imgname):
        path = os.path.join('bg', imgname)
        self.surface = pygame.image.load(path)
        self.imgname = imgname
        self.done = False

    def render(self, screen, dt=0):
        screen.blit(self.surface, (0,0))
        self.done = True

    def process_event(self, event):
        pass

char_pause = {
    '?': 200,
    '!': 200,
    '.': 150,
    ',': 100,
    '(': 50,
    ')': 50,
}
default_pause = 5

text_counter = 0

pause_count = 0
char_num = 0


def blit_text(surface, text, pos, font, color=pygame.Color('white')):
    space = font.get_rect(' ').width  # The width of a space.
    max_width, max_height = surface.get_size()
    x, y = pos
    for line in text.splitlines():
        so_far = ""
        for word in line.split(' '):
            test = so_far + word + " "
            rect = font.get_rect(test)
            if rect.width > max_width-100:
                word_surface, rect = font.render(so_far, color)
                surface.blit(word_surface, (x, y))
                y += rect.h+10
                so_far = word
            else:
                so_far = test

        if len(so_far) > 0:
            word_surface, rect = font.render(so_far, color)
            surface.blit(word_surface, (x, y))
            y += rect.h+10
        # y += word_height  # Start on new row.

class TextView():
    def __init__(self, text):
        self.text = text
        self.char_num = 0
        self.pause_count = 0
        self.done = False
        self.font = pygame.freetype.SysFont("Arial", 30)

    def process_event(self, event):
        if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
            if self.char_num < len(self.text):
                self.char_num = len(self.text)
            else:
                self.done = True

    def render(self, screen, dt=1/60.0):
        s = pygame.Surface((1800,250), pygame.SRCALPHA)   # per-pixel alpha
        s.fill((0,0,0,128))                         # notice the alpha value in the color
        screen.blit(s, (40,780))

        text = self.text[:self.char_num+1]
        blit_text(screen, text, (50,825), self.font)

        self.pause_count -= dt*1.5
        if self.pause_count <= 0 and self.char_num < len(self.text):
            curr_char = self.text[self.char_num]
            self.pause_count = char_pause.get(curr_char, default_pause)
            self.char_num += 1


class ChoiceMenu():
    def __init__(self, choice_id, choices):
        self.choice_id = choice_id
        self.selected = 0
        self.choices = choices

        self.font = pygame.freetype.SysFont("Arial", 30)

        self.done = False

    def process_event(self, event):
        if event.key == pygame.K_UP:
            self.selected = max(self.selected - 1, 0)
        elif event.key == pygame.K_DOWN:
            self.selected = min(self.selected + 1, len(self.choices)-1)
        elif event.key == pygame.K_RETURN:
            self.done = True

    def render(self, screen, dt=1/60.0):
        for i, choice in enumerate(self.choices):
            surf, rect = self.font.render(choice, (255, 255, 255))
            screen.blit(surf, (100, 200 + i*100))
            if self.selected == i:
                r = rect.move((100, 200 + i*100))
                r.y -= r.h
                r = r.inflate(50, 50)
                pygame.draw.rect(screen, (255, 200, 255), r, 5)



class ConditionalView():
    def __init__(self, choice_id, choice_select, view):
        self.choice_id = choice_id
        self.choice_select = choice_select
        self.state = {}
        self.view = view

    def update_state(self, state):
        self.state = state

    def process_event(self, event):
        self.view.process_event(event)

    def render(self, screen, dt=1/60.0):
        if self.state.get(self.choice_id, self.choice_select) != self.choice_select:
            self.done = True
            return
        else:
            self.view.render(screen, dt)
            self.done = self.view.done

class BrainChoice():
    def __init__(self, choice_id):
        self.choice_id = choice_id
        self.state = {}
        self.selected = 0

    def update_state(self, state):
        self.state = state

    def render(self, screen, dt=1/60.0):
        pass
