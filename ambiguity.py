import gym
import numpy as np

import babyai
from babyai.levels.verifier import *
from babyai.levels.levelgen import *


"""
Make the description of an object ambiguous given the object as an ObjDesc
and the environment. Used as a helper function for make_ambiguous().
"""
def make_object_ambiguous(obj, env):
    obj.find_matching_objs(env)

    # assert len(obj.obj_set) > 0, "no object matching description"

    options_to_drop = []

    if obj.type:
        options_to_drop.append('type')
    if obj.color:
        options_to_drop.append('color')
    if obj.loc:
        options_to_drop.append('loc')

    prob_each = 1 / len(options_to_drop)
    dropped = False

    if obj.type:
        # Drop the object type from the instruction
        if np.random.uniform() < prob_each:
            dropped = True
            s = 'object'
        else:
            s = str(obj.type)
    else:
        s = 'object'

    if obj.color:
        # Drop the object color from the instruction
        if not dropped and (np.random.uniform() < prob_each or len(options_to_drop) == 2):
            pass
        else:
            s = obj.color + ' ' + s

    if obj.loc:
        # Drop the object location from the instruction
        if not dropped:
            pass
        else:
            if obj.loc == 'front':
                s = s + ' in front of you'
            elif obj.loc == 'behind':
                s = s + ' behind you'
            else:
                s = s + ' on your ' + obj.loc

    # Singular vs plural
    if len(obj.obj_set) > 1:
        s = 'a ' + s
    else:
        s = 'the ' + s

    return s


"""
Main function to get an ambiguous version of an instruction. Takens
the instruction (subclass of Instr) and the environment, returns a
string representing the ambiguous instruction.

Example usage:

env = gym.make("BabyAI-OpenDoorsOrderN2-v0")
obs = env.reset()
ambiguous_instr = make_ambiguous(env.instrs, env)

TODO:
- Exactly one object or sub-instruction (on the same 'level') is made ambiguous. Need
  to randomize how many are made ambiguous so that task is harder
- Some instructions may not really be ambiguous, remove those as possible outputs.
- "do something with the yellow door" --> probably open
- "open the green object" --> probably door because only doors can be opened (I think?)

"""
def make_ambiguous(instr, env):
    # In case we get to an instruction with multiple ways to make ambiguous
    rand_num = np.random.uniform()

    # We are corrupting a description of an object
    if type(instr) == ObjDesc:
        return make_object_ambiguous(instr, env)
    # We are corrupting an 'open' instruction
    elif type(instr) == OpenInstr:
        # With 0.5 probability, don't specify to open.
        # With 0.5 probability, corrupt the instruction that follows
        return "open " + make_ambiguous(instr.desc, env)
    elif type(instr) == GoToInstr:
        return "go to " + make_ambiguous(instr.desc, env)
    elif type(instr) == PickupInstr:
        return "pick up " + make_ambiguous(instr.desc, env)
    elif type(instr) == PutNextInstr:
        if rand_num < 1/3:
            return 'put ' + make_ambiguous(instr.desc_move, env) + ' next to ' + instr.desc_fixed.surface(env)
        elif rand_num < 2/3:
            return 'put ' + instr.desc_move.surface(env) + ' next to ' + make_ambiguous(instr.desc_fixed, env)
        else:
            return 'put ' + make_ambiguous(instr.desc_move, env) + ' next to ' + make_ambiguous(instr.desc_fixed, env)
    elif type(instr) == BeforeInstr:
        if rand_num < 1/3:
            return make_ambiguous(instr.instr_a, env) + ', then ' + instr.instr_b.surface(env)
        elif rand_num < 2/3:
            return instr.instr_a.surface(env) + ', then ' + make_ambiguous(instr.instr_b, env)
        else:
            return make_ambiguous(instr.instr_a, env) + ', then ' + make_ambiguous(instr.instr_b, env)
    elif type(instr) == AfterInstr:
        if rand_num < 1/3:
            return make_ambiguous(instr.instr_a, env) + ' after you ' + instr.instr_b.surface(env)
        elif rand_num < 2/3:
            return instr.instr_a.surface(env) + ' after you ' + make_ambiguous(instr.instr_b, env)
        else:
            return make_ambiguous(instr.instr_a, env) + ' after you ' + make_ambiguous(instr.instr_b, env)
    elif type(instr) == AndInstr:
        if rand_num < 1/3:
            return make_ambiguous(instr.instr_a, env) + ' and ' + instr.instr_b.surface(env)
        elif rand_num < 2/3:
            return instr.instr_a.surface(env) + ' and ' + make_ambiguous(instr.instr_b, env)
        else:
            return make_ambiguous(instr.instr_a, env) + ' and ' + make_ambiguous(instr.instr_b, env)


"""
Testing function to print ambiguous instructions for all
the environments available on the platform.
"""
def print_ambiguous_instructions():
    # Environments below cause internal errors in BabyAI, so skipped
    to_skip = set(["PutNextS5N2Carrying", "PutNextS6N3Carrying", "PutNextS7N4Carrying"])
    for name in level_dict:
        if name in to_skip: continue
        print(name)
        for i in range(3):
            env = gym.make(f"BabyAI-{name}-v0")
            obs = env.reset()

            print(f"Original instruction: {obs['mission']}")
            for i in range(3):
                ambiguous_instr = make_ambiguous(env.instrs, env)
                print(f"Ambiguous instruction {i}: {ambiguous_instr}")

            print()
        print()


if __name__ == '__main__':
    print_ambiguous_instructions()
