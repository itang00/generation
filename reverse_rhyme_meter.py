import string
import re
import sys
import numpy as np

# Load the sonnets, removing blank lines and header (sonnet #) lines.
lines = []
with open('data/shakespeare.txt') as fin:
    counter = 0 # assure that all sonnets have 14 lines
    for l in fin:
        l = l.strip()
        if not l:
            continue
        if l[0] in string.digits:
            assert counter%14 == 0
            counter = 0
        else:
            lines.append(l)
assert len(lines)%14 == 0

# Parse the lines and map them to integers.
obs_map = {}
obs_seqs = []
for l in lines:
    # Break into words, treating punctuation (besides ') as separate words.
    words = re.findall(r"[\w\-']+|[.,!?;:]", l)
    assert words

    # Remove case information for the first word in each line, since they are
    # all capitalized and so the capitalization has no semantic meaning.
    words[0] = words[0].lower()

    # The ' character is used both as a quoting device, and as a prefix. We
    # strip it here because we can't really tell how it's being used. We can
    # recover the prefix uses later from the syllable dictionary.
    words = [w.strip("'()") for w in words]

    # Register words and convert the line to an integer sequence.
    seq = []
    for w in words:
        if w not in obs_map:
            obs_map[w] = len(obs_map)
        seq.append(obs_map[w])
    obs_seqs.append(seq)

# Load the syllable map. Structure:
# syl_map[word] -> {
#   False: [possible syl counts when not at end of line]
#   True:  [possible syl counts when last word in line]}
syl_map = {}
with open('data/Syllable_dictionary.txt') as fin:
    for line in fin:
        spec = line.strip().split()
        w = spec[0]
        c = {False: [], True: []}
        for f in spec[1:]:
            c['E' in f].append(int(f[-1]))
        c[True] += c[False]
        syl_map[w] = c

# Fix discrepancies between syl_map and obs_map.
for w in list(obs_map.keys()):
    if w not in syl_map:
        # Copy syllable counts for capitalization differences.
        if w.lower() in syl_map:
            syl_map[w] = syl_map[w.lower()]
        # Recover ' prefixes.
        elif ("'" + w) in syl_map:
            obs_map["'" + w] = obs_map[w]
            del obs_map[w]
        # Recover ' postfixes.
        elif (w + "'") in syl_map:
                obs_map[w + "'"] = obs_map[w]
                del obs_map[w]
        # Handle punctuation.
        elif w in '.,!?;:':
            syl_map[w] = {False: [0], True: [0]}
        else:
            sys.stderr.write(f'Not in syl_map: {w}\n')
            assert False
for w in list(syl_map.keys()):
    if w not in obs_map:
        del syl_map[w]

# Detect the possible stresses of each word by looking at the syllable
# offsets that it can occur at in a line. For words with multiple possible
# syllable counts, this is a little tricky. When there's only one variable word
# in a line, we can infer the syllable count being used by counting syllables.
# Structure:
# stress_map[word][syl count] -> [even possible, odd possible]
stress_map = {
        w: {s: [False, False] for s in syl_map[w][True]} for w in obs_map}
obs_rev = {obs_map[w]: w for w in obs_map}
for seq in obs_seqs:
    off = 0
    seqlen = len(seq)
    if obs_rev[seq[-1]] in '.,!?;:':
        seqlen -= 1
    for i, x in enumerate(seq):
        w = obs_rev[x]
        sc_list = syl_map[w][i == seqlen-1]
        if len(sc_list) == 1:
            sc = sc_list[0]
        else:
            tot = off
            bad = False
            for j, y in enumerate(seq[i+1:]):
                scl = syl_map[obs_rev[y]][i+1+j == seqlen-1]
                if len(scl) > 1:
                    bad = True
                    break
                tot += scl[0]
            if bad:
                break
            sc = 10-tot
        try: # some (~3?) lines just don't have 10 syllables...
            stress_map[w][sc][off%2] = True
        except:
            print('Err: ', w, sc, off)
            break
        off += sc

# Make capitalization pairs share stress info.
for w in stress_map:
    if (w != w.lower()) and (w.lower() in stress_map):
        l = w.lower()
        for s in stress_map[w]:
            stress_map[w][s][0] |= stress_map[l][s][0]
            stress_map[w][s][1] |= stress_map[l][s][1]
        stress_map[l] = stress_map[w]

# Remove nonsense ending counts (that are disallowed by stress info).
for w in syl_map:
    syl_map[w][True] = [c for c in syl_map[w][True] if stress_map[w][c][0]]

# Discover rhyming pairs.
pairs = set()
for i in range(0, len(obs_seqs), 14):
    get_last = lambda j: (obs_seqs[j][-1]
            if obs_rev[obs_seqs[j][-1]] not in '.,!?;:'
            else obs_seqs[j][-2])
    def addp(a, b):
        pairs.add(tuple(sorted((get_last(i+a), get_last(i+b)))))
    for p in [(0,2), (1,3), (4,6), (5,7), (8,10), (9,11), (12,13)]:
        addp(*p)

# Generate a poem from a trained HMM.
A = np.exp(np.load(sys.argv[1]))
O = np.exp(np.load(sys.argv[2]))
L, D = O.shape


rhyme_masks = {}
rhymable_mask = np.zeros(D)
for p in list(pairs):
    for i in (0,1):
        if p[i] not in rhyme_masks:
            rhyme_masks[p[i]] = np.zeros(D)
        rhyme_masks[p[i]][p[(i+1)%2]] = 1
        rhymable_mask[p[i]] = 1
rhyme_masks[-1] = rhymable_mask

# Create a table allowing for the selection of valid words for reverse-order
# placement given the current (reverse) syllable offset.
syl_rev = {obs_map[w]: syl_map[w] for w in obs_map}
stress_mask = {0: np.zeros(D), 1: np.zeros(D)}
stress_syls = {}
for w in stress_map:
    sclist = syl_map[w][False]
    x = obs_map[w]
    stress_syls[x] = {0: [], 1: []}
    for sc in sclist:
        for i in (0,1):
            if stress_map[w][sc][i]:
                stress_mask[(sc+i)%2][x] = 1
                stress_syls[x][(sc+i)%2].append(sc)

def gen_possible_lines(word0, state0, p0):
    q = []
    for sc in syl_rev[word0][True]:
        q.append(([word0], sc, state0, p0))
    #hs0p = O[,start]/np.sum(O[,start])
    #for hs0 in np.random.choice(np.arange(L), 2, replace=False, p=hs0p):
        #for sc in syl_rev[start][True]:
            #q.append(([start], sc, hs0, p0*hs0p[hs0]))
    good_lines = []
    while q and len(good_lines) < 2000:
        cur = q.pop()
        seq, sylc, state, prob = cur
        #print(seq, len(good_lines))
        if sylc >= 10:
            if sylc == 10:
                good_lines.append(cur)
            continue
        for next_state in np.random.choice(
                np.arange(L), 2, replace=False, p=A[state,:]):
            wgt = O[next_state,:]*stress_mask[sylc%2]
            wgt /= np.sum(wgt)
            for next_word in np.random.choice(
                    np.arange(D), 2, replace=False, p=wgt):
                if (obs_rev[next_word] in '.,!?;:'
                        and obs_rev[seq[-1]] in '.,!?;:'):
                    continue
                for sc in stress_syls[next_word][sylc%2]:
                    q.append((
                        seq + [next_word],
                        sylc + sc,
                        next_state,
                        prob * A[state,next_state] * wgt[next_word]))
    return good_lines

def gen_next_line(prevstate, rhymeword):
    lines = []
    for s0 in np.random.choice(np.arange(L), 2, replace=False, p=A[prevstate,:]):
        #wgt = O[s0,:]*rhyme_masks[rhymeword]
        wgt = rhyme_masks[rhymeword]
        wgt /= np.sum(wgt)
        for w0 in np.random.choice(np.arange(D), 1, replace=False, p=wgt):
            lines += gen_possible_lines(w0, s0, A[prevstate,s0]*wgt[w0])
            print(len(lines))
    cand_w = np.array([c[3] for c in lines])
    choice = np.random.choice(np.arange(len(lines)), p=cand_w/np.sum(cand_w))
    return lines[choice]

#def gnl_safe(prevstate, rhymeword)

def gen_complex():
    lines = []
    # generate the first (last) line
    candidates = []
    for w0 in np.random.choice(np.arange(D), 2, replace=False,
            p=rhymable_mask/np.sum(rhymable_mask)):
        hs0p = O[:,w0]/np.sum(O[:,w0])
        for hs0 in np.random.choice(np.arange(L), 2, replace=False, p=hs0p):
            candidates += gen_possible_lines(w0, hs0, hs0p[hs0])
    cand_w = np.array([c[3] for c in candidates])
    choice = np.random.choice(np.arange(len(candidates)),
            p=cand_w/np.sum(cand_w))
    lines.append(candidates[choice])
    # generate other lines
    def addnext(rhyme):
        if rhyme != -1:
            rhyme = lines[rhyme][0][0]
        lines.append(gen_next_line(lines[-1][2], rhyme))
    addnext(0)  # line 13 (@1)
    addnext(-1) # line 12 (@2)
    addnext(-1) # line 11 (@3)
    addnext(2)  # line 10 (@4)  (rhymes with 12)
    addnext(3)  # line 9  (@5)  (rhymes with 11)
    addnext(-1) # line 8  (@6)
    addnext(-1) # line 7  (@7)
    addnext(6)  # line 6  (@8)  (rhymes with 8)
    addnext(7)  # line 5  (@9)  (rhymes with 7)
    addnext(-1) # line 4  (@10)
    addnext(-1) # line 3  (@11)
    addnext(10) # line 2  (@12) (rhymes with 4)
    addnext(11) # line 1  (@13) (rhymes with 3)
    print(lines)
    return [l[0] for l in lines]

def gen_complex_safe():
    #return gen_complex()
    for _ in range(50):
        try:
            return gen_complex()
        except Exception as exc:
            print(exc)

def to_text(poem):
    text = ''
    for line in reversed(poem):
        capnext = True
        for i, x in enumerate(reversed(line)):
            w = obs_rev[x]
            if i == 0 and w in '.,!?;:':
                continue
            if w not in '.,!?;:' and i != 0:
               text += ' '
            if capnext:
                text += w.capitalize()
                capnext = False
            else:
                text += w
            if w in '.!?;:':
                capnext = True
        text += '\n'
    return text

sys.stdout.write(to_text(gen_complex_safe()))
