import numpy as np
import gymnasium as gym

env = gym.make('CartPole-v1')
nActions = env.action_space.n
upperBounds = env.observation_space.high
lowerBounds = env.observation_space.low
cartVelHigh = 5
cartVelLow = -5
poleVelHigh = 10
poleVelLow = -10
upperBounds[1]=cartVelHigh
lowerBounds[1]=cartVelLow
upperBounds[3]=poleVelHigh
lowerBounds[3]=poleVelLow

nBinsCartPos = 30
nBinsCartVel = 30
nBinsPoleAngle = 50
nBinsPoleVel = 50
nBins = [nBinsCartPos, nBinsCartVel, nBinsPoleAngle, nBinsPoleVel]

learnRate = 0.1
discount = 0.95
epsilon = 1
nEps = 60001

qMat = np.random.uniform(low=0, high=1, size=(nBins[0], nBins[1], nBins[2], nBins[3], nActions))

def getStateIndex(state):
    cartPosBin = np.linspace(lowerBounds[0], upperBounds[0], nBins[0])
    cartVelBin = np.linspace(lowerBounds[1], upperBounds[1], nBins[1])
    poleAngleBin = np.linspace(lowerBounds[2], upperBounds[2], nBins[2])
    poleVelBin = np.linspace(lowerBounds[3], upperBounds[3], nBins[3])

    cartPosInd = np.maximum(np.digitize(state[0], cartPosBin) - 1, 0)
    cartVelInd = np.maximum(np.digitize(state[1], cartVelBin) - 1, 0)
    poleAngleInd = np.maximum(np.digitize(state[2], poleAngleBin) - 1, 0)
    poleVelInd = np.maximum(np.digitize(state[3], poleVelBin) - 1, 0)

    return (cartPosInd, cartVelInd, poleAngleInd, poleVelInd)

def selectAction(state, epInd, epsilon):

    if epInd < 2000 :
        return np.random.choice(nActions)
    
    randNum = np.random.random()

    if randNum < epsilon :
        return np.random.choice(nActions)
    else :
        stInd = getStateIndex(state)
        return np.random.choice(np.where(qMat[stInd] == np.max(qMat[stInd]))[0])
    
prevRew = []
prevReward = 0

for epInd in range(nEps) :
    epRewards = []

    (state, _) = env.reset()
    state = list(state)

    terminalState = False

    while not terminalState :
        stInd = getStateIndex(state)
            
        action = selectAction(state, epInd, epsilon)

        (statePr, reward, terminalState, _, _) = env.step(action)

        epRewards.append(reward)

        statePr = list(statePr)
        stPrInd = getStateIndex(statePr)

        qPrMax = np.max(qMat[stPrInd])

        if not terminalState :
            valChange = reward + discount * qPrMax - qMat[stInd + (action,)]
            qMat[stInd + (action,)] = qMat[stInd + (action,)] + learnRate * valChange
        else :
            valChange = reward - qMat[stInd + (action,)]
            qMat[stInd + (action,)] = qMat[stInd + (action,)] + learnRate * valChange

        state = statePr
    
    sumOfRewards = np.sum(epRewards)
    if epInd > 5000 and prevReward < sumOfRewards:
        epsilon = 0.99995 * epsilon
    prevReward = sumOfRewards
    prevRew.append(sumOfRewards)
    if epInd % 2000 == 0 :
        print("Episode " + str(epInd) + " Avg. Reward = " + str(np.sum(prevRew)/2000) + " epsilon = " + str(epsilon))
        prevRew = []

env.close()

print("Training Done.")

env = gym.make('CartPole-v1', render_mode='human')
(state, _)=env.reset()
state = list(state)

terminalState = False
finalReward = 0

while not terminalState :
    stInd = getStateIndex(state)
    action = np.random.choice(np.where(qMat[stInd] == np.max(qMat[stInd]))[0])

    (statePr, reward, terminalState, _, _) = env.step(action)
    env.render()
    finalReward += reward
    state = list(statePr)

print(finalReward)
env.close()