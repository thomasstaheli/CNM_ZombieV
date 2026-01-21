# ZombieV
2D top down zombie shooter game in C++ using SFML as graphics library and custom game engine (Ligths, Physics, Entity creation, etc...)

## Example
![Zombie](https://github.com/johnBuffer/ZombieV/blob/master/img/illustration.png)

### Video link

 - [Single player](https://www.youtube.com/watch?v=pj3m3Fu3i5A)
 - [Bots](https://www.youtube.com/watch?v=LflP2BUqJQc)
 - [Lights](https://www.youtube.com/watch?v=rCyaakRHUJ0)

## Run demo

In the release folder you will find binaries, each one corresponding to a scenario
*  Solo game
*  Game with bots
*  Lights demo
*  Bots + night

## Compile program

To run this programm you will need :

- C++ compilator with C++17 or above
- Cmake version 3.10 or above
- SFML library

From root, base of the repository :

```bash
cd build
# It will build and launch the game
../run.sh
```

To quit the game, you can press escape (ESC) or you can click on the cross on the top right of the window.

## Profiling

Each run, there is a file, called `mresures_fps.csv` that save the differents measures of the FPS.
If you want to see the differents measure, you can display it with the python script from root.

You will need, `python3` with the librairies `panda` and `matplotlib`.

```bash
python3 plot_mesure.py
```

## Fonctions that can be parallelize

- `void Zombie::_getTarget()` in Zombie.cpp
- `void Zombie::update(GameWorld& world)` in Zombie.cpp
- `void Bot::getTarget(GameWorld* world)` in Bot.cpp
- `void Bot::computeControls(GameWorld& world)` in Bot.cpp
- `void Bot::update(GameWorld& world)` in Bot.cpp
