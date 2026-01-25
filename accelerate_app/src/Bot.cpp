#include <limits> // Pour float max
#include <omp.h>

#include "Bot.hpp"
#include "System/GameWorld.hpp"
#include "Zombie.hpp"

// Une petite structure pour stocker le résultat d'un thread
struct SearchResult {
    float minDist = std::numeric_limits<float>::max();
    Zombie* target = nullptr;
};

Bot::Bot() :
    HunterBase(0, 0),
    m_target(nullptr)
{

}

Bot::Bot(float x, float y) :
    HunterBase(x, y),
    m_target(nullptr),
    m_getTargetCount(0)
{
    Light light;
    light.color = sf::Color(255, 127, 0);
    light.intensity = 1.0f;
    light.radius  = 0;
    _shootLight = GameRender::getLightEngine().addDurableLight(light);

}

void Bot::update(GameWorld& world)
{
    computeControls(world);

    _currentWeapon->update();
    _time += DT;

    if (_state == SHOOTING)
    {
        if (!_currentWeapon->fire(&world, this))
        {
            _changeState(IDLE);
            if (_currentWeapon->isMagEmpty())
            {
                _changeAnimation(_currentWeapon->getReloadAnimation(), false);
                _state = RELOADING;
            }
        }
    }

    if (_state == RELOADING && _currentAnimation.isDone())
    {
        _currentWeapon->reload();
        _state = IDLE;
    }

    _shootLight->radius = 0;
    if (_state == SHOOTING)
    {
        bool wait = _lastState==SHOOTING;
        _changeAnimation(_currentWeapon->getShootAnimation(), wait);
        _shootLight->radius = 350;
    }
    else if (_state == MOVING)
    {
        bool wait = !(_lastState==IDLE);
        _changeAnimation(_currentWeapon->getMoveAnimation(), wait);
    }
    else
    {
        _changeAnimation(_currentWeapon->getIdleAnimation());
    }

    _shootLight->position = _currentWeapon->getFireOutPosition(this);
    /*_flashlight->position = _shootLight->position;
    _littleLight->position = _shootLight->position;
    _flashlight->angle = getAngle()+PI;*/
}

void Bot::computeControls(GameWorld& world)
{
    if (m_target)
    {
        Vec2 vTarget(m_target->getCoord(), getCoord());
        Vec2 direction(cos(_angle), sin(_angle));
        Vec2 directionNormal(-direction.y, direction.x);

        float dist = vTarget.getNorm();
        float vx = vTarget.x/dist;
        float vy = vTarget.y/dist;

        float dot2 = vx*directionNormal.x + vy*directionNormal.y;
        float coeff = 0.25f;

        float absDot = std::abs(dot2);
        coeff *= absDot;

        if (absDot<0.25f || dist < 100)
        {
            if (dist < 300)
            {
                _changeState(SHOOTING);
                if (dist < 100)
                {
                    _feetTime += DT;
                    move(vx*_speed, vy*_speed);
                }
            }
            else
            {
                float speedFactor = 0.25f;
                move(-vx*_speed*speedFactor, -vy*_speed*speedFactor);
                _feetTime += DT*speedFactor;
            }
        }
        else
        {
            _changeState(IDLE);
        }

        _angle += dot2>0?-coeff:coeff;

        if (m_target->isDying())
            m_target = nullptr;
    }
    else
    {
        _changeState(IDLE);
        getTarget(&world);
    }

    m_coord = getBodyCoord();
}

void Bot::getTarget(GameWorld* world)
{
    ++m_getTargetCount;

    // Préparation des données
    // Comme indiqué dans le rapport, cette étape est très lente et elle ralenti fortement processus
    // le fait d'applatir les données, cela gâche nos performances
    // Une solution proposée, serait de revoir le système complet de stockage des entités
    // en évitant à tout pris la Pool d'objet (liste) et en utilisant des vecteurs pour chaque type d'entité
    std::vector<Zombie*> zombies;
    zombies.reserve(10000);
    Zombie* z = nullptr;
    // On ne peut pas paraléliser cette étape
    while (Zombie::getNext(z)) {
        zombies.push_back(z);
    }

    int nZombies = zombies.size();
    if (nZombies == 0) return;

    Vec2 myPos = getCoord();

    // On détermine combien de threads on veut utiliser AU MAXIMUM
    // On véite d'utiliser tous les threads (sur le PC de test => 20 threads max)
    int desiredThreads = 10; 
    // On récupère le max possible sur la machine pour ne pas demander l'impossible
    int hardwareMax = omp_get_max_threads();
    int actualThreads = std::min(desiredThreads, hardwareMax);

    // On crée le tableau de résultats avec la taille exacte du nombre de threads qu'on va lancer
    std::vector<SearchResult> threadResults(actualThreads);
    
    // On sépare les zombies, on chaque thread calcule son minimum local
    #pragma omp parallel num_threads(actualThreads)
    {
        int threadID = omp_get_thread_num();
        
        // Initialisation locale
        float localMinDist = std::numeric_limits<float>::max();
        Zombie* localTarget = nullptr;

        // On laisse OpenMP découper la boucle complète (0 à nZombies)
        // C'est beaucoup plus sûr que le calcul manuel
        // #pragma omp for nowait
        for (int i = 0; i < nZombies; ++i)
        {
            Zombie* currentZombie = zombies[i];
            Vec2 zPos = currentZombie->getCoord();
            
            float dx = zPos.x - myPos.x;
            float dy = zPos.y - myPos.y;
            float distSq = dx*dx + dy*dy;

            if (distSq < localMinDist) {
                localMinDist = distSq;
                localTarget = currentZombie;
            }
        }

        // Écriture sécurisée car threadID < actualThreads
        threadResults[threadID].minDist = localMinDist;
        threadResults[threadID].target = localTarget;
    }

    // Barrière implicite ici

    // Maintenant que tous les threads ont fini leur travail
    float globalMinDist = std::numeric_limits<float>::max();
    Zombie* globalTarget = nullptr;

    // On parcourt tous les résultats afin de déterminer le minimum des minimums locaux des threads
    for (int i = 0; i < actualThreads; ++i)
    {
        if (threadResults[i].target != nullptr && threadResults[i].minDist < globalMinDist)
        {
            globalMinDist = threadResults[i].minDist;
            globalTarget = threadResults[i].target;
        }
    }

    if (globalTarget)
    {
        m_target = globalTarget;
    }
}

/* OLD
void Bot::getTarget(GameWorld* world)
{
    ++m_getTargetCount;
    Zombie* zombie = nullptr;
    Zombie* target = nullptr;
    float minDist  = -1;

    int i = 0;

    // Parcourt tous les zombies -> très lourd si le nombre de bot est élevé
    while (Zombie::getNext(zombie))
    {
        Vec2 v(zombie->getCoord(), getCoord());
        float dist = v.getNorm2();

        if (dist < minDist || minDist < 0)
        {
            minDist = dist;
            target = zombie;
        }
    }

    if (target)
    {
        m_target = target;
    }
}
*/

void Bot::hit(WorldEntity* entity, GameWorld* gameWorld)
{
    switch(entity->getType())
    {
        case(EntityTypes::ZOMBIE) :
        {
            m_target = entity;
            break;
        }
        default:
            break;
    }
}

void Bot::initialize()
{
    //HunterBase::init();
}

