import pygame
import pygame.freetype  # Import the freetype module.
from views import Background, ChoiceMenu, ConditionalView, TextView

# from pylsl import StreamInlet, resolve_stream

# # first resolve an EEG stream on the lab network
# def get_inlet():
#     print("looking for an EEG stream...")
#     streams = resolve_stream('type', 'EEG')
#     print("Streams", streams)
#     for stream in streams:
#         print("id", str(stream.source_id()))
#         if stream.source_id() == "stressdata":
#             inlet = StreamInlet(stream)
#     return inlet


# inlet = get_inlet()

pygame.init()
screen = pygame.display.set_mode((1920, 1080))
running = True


clock = pygame.time.Clock()


ALL_CHOICES = {
    'plant': ChoiceMenu("plant", ["Don't water the plant, it's too late",
                                  "Water the plant"])
}

def read_script(filename):
    all_texts = []
    row = ""
    with open(filename, 'r') as f:
        for line in f:
            if line[:3] == "!bg":
                imgname = line.strip().split(' ')[1]
                all_texts.append(Background(imgname))
            elif line[:7] == "!choice":
                choice = line.strip().split(' ')[1]
                all_texts.append(ALL_CHOICES[choice])
            elif line[:5] == "!cond":
                L = line.strip().split(' ')
                choice_id = L[1]
                choice_select = int(L[2])
                text = ' '.join(L[3:])
                v = ConditionalView(choice_id, choice_select, TextView(text))
                all_texts.append(v)
            elif line.strip() == "":
                t = row.strip()
                if t != '':
                    all_texts.append(TextView(t))
                row = ""
            else:
                row += line
    if len(row.strip()) > 0:
        all_texts.append(TextView(row.strip()))
    return all_texts

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
