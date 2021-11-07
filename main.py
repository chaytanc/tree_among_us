import pygame
import pygame.freetype  # Import the freetype module.
from views import Background, ChoiceMenu, ConditionalView

pygame.init()
screen = pygame.display.set_mode((1920, 1080))
running = True


clock = pygame.time.Clock()

from utils import read_script

slides = read_script('script.txt')
# script commands
# !bg blah.jpg
# !choice plant
# anything else is text

background = Background("black.jpg")

slide_ix = 0
slide = slides[slide_ix]
if isinstance(slide, Background):
    background = slide

choice_states = { }

while running:
    dt = clock.tick(90)

    if isinstance(slide, ConditionalView):
        slide.update_state(choice_states)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            else:
                slide.process_event(event)

    screen.fill((40, 40, 40))
    background.render(screen, dt)
    slide.render(screen, dt)
    pygame.display.flip()

    if slide.done:
        if isinstance(slide, ChoiceMenu):
            choice_states[slide.choice_id] = slide.selected
            print(choice_states)
        slide_ix += 1
        slide = slides[slide_ix]
        if isinstance(slide, Background):
            background = slide
pygame.quit()
