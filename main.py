import pygame
import pygame.freetype  # Import the freetype module.
from views import Background, ChoiceMenu, ConditionalView, TextView, BrainChoice

from pylsl import StreamInlet, resolve_stream


# first resolve an EEG stream on the lab network
def get_inlet():
    print("looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    print("Streams", streams)
    for stream in streams:
        print("id", str(stream.source_id()))
        inlet = None
        while(inlet == None):
            if stream.source_id() == "stressdata":
                inlet = StreamInlet(stream)
    return inlet

# inlet = get_inlet()


pygame.init()
screen = pygame.display.set_mode((1920, 1080))
running = True

clock = pygame.time.Clock()

ALL_CHOICES = {
    'plant': ChoiceMenu("plant", ["Don't water the plants",
                                  "Water the plants"]),
    'sparekill': ChoiceMenu("sparekill", ["Spare your friend", "Kill him"]),
    'stress': ChoiceMenu("stress", ["No it's chill between us", "Yeah you stress me out"])
}


def parse_line(line):
    line = line.strip()
    if len(line) > 0 and line[0] == '#':
        return ""
    if line[:3] == "!bg":
        imgname = line.split(' ')[1]
        return Background(imgname)
    elif line[:7] == "!choice":
        choice = line.split(' ')[1]
        return ALL_CHOICES[choice]
    elif line[:5] == "!cond":
        L = line.split(' ')
        choice_id = L[1]
        choice_select = int(L[2])
        text = ' '.join(L[3:])
        c = parse_line(text)
        if isinstance(c, str):
            c = TextView(c)
        return ConditionalView(choice_id, choice_select, c)
    elif line[:6] == "!brain":
        L = line.split(' ')
        choice_id = L[1]
        return BrainChoice(choice_id)
    else:
        return line


def read_script(filename):
    all_texts = []
    row = ""
    with open(filename, 'r') as f:
        for line in f:
            c = parse_line(line)
            if isinstance(c, str):
                if c == "":
                    t = row.strip()
                    if t != "":
                        all_texts.append(TextView(t))
                    row = ""
                else:
                    row += line
            else:
                all_texts.append(c)

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

sample = [1, 1, 1]

while running:
    dt = clock.tick(90)

    # c_sample, c_timestamp = inlet.pull_sample(timeout=0)
    # if c_sample is not None:
    #     sample, timestamp = c_sample, c_timestamp

    if isinstance(slide, ConditionalView) or \
       isinstance(slide, BrainChoice):
        slide.update_state(choice_states)

    if isinstance(slide, BrainChoice):
        slide.update_eeg_sample(sample)
        choice_states[slide.choice_id] = slide.selected

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break
            else:
                slide.process_event(event)

    s = sample[1]
    c = s*200

    # screen.fill((0+c, 0+c, 0+c))
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
