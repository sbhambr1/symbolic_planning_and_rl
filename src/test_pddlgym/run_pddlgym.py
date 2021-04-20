import pddlgym
from pddlgym_planners.fd import FD
from pddlgym_planners.ff import FF 
import cv2

def main():
    # See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
    # env = pddlgym.make("PDDLEnvSokoban-v0")
    # env = pddlgym.make("PDDLEnvBlocks-v0")
    env = pddlgym.make("PDDLEnvBlocks-v0")
    
    obs, debug_info = env.reset()
    img = env.render()

    planner = FF() # FD() or FF()
    plan = planner(env.domain, obs)
    for act in plan:
        print("Obs:", obs)
        print("Act:", act)
        obs, reward, done, info = env.step(act)
        img = env.render()
        
        cv2.imshow('frame', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        
    print("Final obs, reward, done:", obs, reward, done)


if __name__ == "__main__":
    main()

