#include "Zombie.hpp"
#include "System/GameWorld.hpp"
#include "System/GameRender.hpp"
#include "Props/Props.hpp"
#include "Bot.hpp"
#include "System/Utils.hpp"
#include "Hunter.hpp"
#include "UnitedEngine/U_2DCollisionManager.h"  // Pour GridCell et phyManager

#include <iostream>
#include <iostream>
#include <cfloat>  // Pour FLT_MAX
#include <cmath>   // Pour sqrt

uint64_t      Zombie::_moveTextureID;
uint64_t      Zombie::_attackTextureID;
Animation   Zombie::_moveAnimation(3, 6, 288, 311, 17, 20);
Animation   Zombie::_attackAnimation(3, 3, 954/3, 882/3, 9, 20);
const float Zombie::TARGET_SEARCH_INTERVAL = 0.5f;  // Recherche tous les 0.5 secondes

Zombie::Zombie() :
    StandardEntity(),
    _targetSearchCooldown(0.0f)
{

}

Zombie::Zombie(float x, float y) :
    StandardEntity(x, y, 0.0f),
    _vertexArray(sf::VertexArray(sf::Quads, 4)),
    _targetSearchCooldown(0.0f)  // Initialiser le cooldown
{
    _speed = 500;
    _life  = 100;
    _done  = false;

    _currentAnimation = _moveAnimation;

    _time = getRandUnder(100.0f);
    _type = EntityTypes::ZOMBIE;

    _currentState = IDLE;
    _marked = false;
    _target = ENTITY_NULL;
}

Zombie::~Zombie()
{
}

void Zombie::kill(GameWorld* world)
{
    world->removeBody(m_bodyID);
    this->remove();
}

void Zombie::setTarget(EntityID target)
{
    _currentState = MOVING;
    _target = target;
}

void Zombie::update(GameWorld& world)
{
    _time += DT;

    if (_target)
    {
        WorldEntity* target = world.getEntityByID(_target);

        if (target)
        {
            Vec2 vTarget(target->getCoord(), m_coord);
            Vec2 direction(cos(_angle), sin(_angle));
            Vec2 directionNormal(-direction.y, direction.x);

            // Optimisation: Utiliser getNorm2() au lieu de getNorm() + sqrt
            float dist2 = vTarget.getNorm2();
            float dist = sqrt(dist2);  // Calculé une seule fois
            
            // Vérifier la division par zéro
            if (dist > 0.0001f) {
                float vx = vTarget.x / dist;
                float vy = vTarget.y / dist;

                float dot2 = vx * directionNormal.x + vy * directionNormal.y;
                float coeff = 0.04f;

                float absDot = std::abs(dot2);
                coeff *= absDot;

                _angle += dot2 > 0 ? -coeff : coeff;

                if (_currentState == MOVING)
                {
                    move(_speed * direction.x, _speed * direction.y);
                }
                else if (_currentAnimation.isDone())
                {
                    // Comparaison sans sqrt: utiliser dist2 pour éviter le sqrt
                    const float ATTACK_RANGE = 3.0f * CELL_SIZE;
                    const float ATTACK_RANGE2 = ATTACK_RANGE * ATTACK_RANGE;
                    
                    if (dist2 < ATTACK_RANGE2)
                    {
                        target->addLife(-5);
                        world.addEntity(ExplosionProvider::getBase(target->getCoord()));
                    }
                }

                if (target->isDying())
                    _target = ENTITY_NULL;
            }
        }
        else
        {
            // Target invalide, rechercher un nouveau
            _target = ENTITY_NULL;
            _targetSearchCooldown = 0.0f;
        }
    }
    else
    {
        // Recherche de target avec cooldown (optimisation clé!)
        _targetSearchCooldown -= DT;
        if (_targetSearchCooldown <= 0.0f)
        {
            _getTargetOptimized(world);  // Utiliser la version optimisée
            _targetSearchCooldown = TARGET_SEARCH_INTERVAL;  // Attendre avant la prochaine recherche
        }
    }

    if (_life < 0)
    {
        _done = true;
    }

    if (_currentAnimation.isDone())
    {
        _currentAnimation = _moveAnimation;
        _currentState = MOVING;
    }

    m_coord = getBodyCoord();
}

/**
 * VERSION ORIGINALE (Conservée pour compatibilité)
 * Scanne TOUS les hunters - O(n) - LENT
 */
void Zombie::_getTarget()
{
    Hunter*  hunter = nullptr;
    EntityID target = ENTITY_NULL;
    float minDist  = -1;

    while (Hunter::getNext(hunter))
    {
        Vec2 v(hunter->getCoord(), m_coord);
        float dist = v.getNorm2();

        if ((dist < minDist || minDist < 0) && !hunter->isDying())
        {
            minDist = dist;
            target = hunter->getID();
        }
    }

    if (target)
    {
        setTarget(target);
    }
}

/**
 * VERSION OPTIMISÉE - NOUVELLE
 * Utilise la grille spatiale pour chercher seulement les hunters proches
 * 
 * Gain de performance:
 * - Avant: Scanne TOUS les hunters (peut être 1000+)
 * - Après: Scanne seulement ~9 cellules de grille = ~50 zombies max par cellule
 * 
 */
void Zombie::_getTargetOptimized(GameWorld& world)
{
    // 1. Récupérer le collisionManager
    U_2DCollisionManager& phyManager = world.getPhyManager();
    
    // 2. Déterminer la position et paramètres de la recherche
    Vec2 zombiePos = m_coord;
    EntityID closestTarget = ENTITY_NULL;
    float closestDist2 = FLT_MAX;  // Distance au carré du plus proche
    
    // 3. PARAMÈTRES DE RECHERCHE
    // La grille utilise CELL_SIZE comme unité (voir U_2DCollisionManager)
    float cellSize = phyManager.getBodyRadius() * 2.0f;  // CELL_SIZE équivalent
    
    // Rayon de recherche: 3 cellules = couverture 9 cellules (3x3)
    const int SEARCH_RADIUS = 1;  // -1 à +1 en x et y = 3x3 cellules
    
    // 4. BOUCLE DE RECHERCHE SUR LA GRILLE
    // Au lieu de chercher dans TOUS les hunters, on cherche seulement
    // dans les 9 cellules voisines
    #pragma omp parallel for collapse(2) reduction(min:closestDist2) shared(closestTarget)
    for (int dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx++)
    {
        for (int dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy++)
        {
            // Calculer la position de la cellule voisine
            Vec2 checkPos = zombiePos + Vec2(dx * cellSize, dy * cellSize);
            
            // Récupérer les corps physiques dans cette cellule
            GridCell* cell = world.getBodiesAt(checkPos);
            
            if (cell)
            {
                // 5. SCANNER LES CORPS DANS CETTE CELLULE
                for (int i = 0; i < cell->_maxIndex; i++)
                {
                    U2DBody_ptr body = cell->_bodies[i];
                    if (!body) continue;
                    
                    // Récupérer l'entité associée au corps physique
                    WorldEntity* entity = body->getEntity();
                    if (!entity) continue;
                    
                    // Vérifier que c'est un Hunter
                    if (entity->getType() != EntityTypes::HUNTER) continue;
                    
                    // Vérifier que le Hunter n'est pas en train de mourir
                    Hunter* hunter = static_cast<Hunter*>(entity);
                    if (hunter->isDying()) continue;
                    
                    // 6. CALCULER LA DISTANCE AU CARRÉ (sans sqrt!)
                    Vec2 delta = hunter->getCoord() - zombiePos;
                    float dist2 = delta.x * delta.x + delta.y * delta.y;
                    
                    // 7. METTRE À JOUR LE PLUS PROCHE SI NÉCESSAIRE
                    if (dist2 < closestDist2)
                    {
                        closestDist2 = dist2;
                        closestTarget = hunter->getID();
                    }
                }
            }
        }
    }
    
    // 8. SI UN TARGET A ÉTÉ TROUVÉ
    if (closestTarget != ENTITY_NULL)
    {
        setTarget(closestTarget);
    }
}

void Zombie::render()
{
    if (GameRender::isVisible(this))
    {
        const Vec2& coord = m_coord;
        float x = coord.x;
        float y = coord.y;

        GraphicUtils::initQuad(_vertexArray, sf::Vector2f(288, 311), sf::Vector2f(144, 155), SCALE*0.25);
        GraphicUtils::transform(_vertexArray, sf::Vector2f(x, y), _angle);

        _currentAnimation.applyOnQuad(_vertexArray, _time);

        GameRender::addQuad(_currentAnimation.getTexture(), _vertexArray, RenderLayer::RENDER);
        GameRender::addShadowCaster(getCoord(), CELL_SIZE);
        GraphicUtils::createEntityShadow(this);
    }
}

void Zombie::initialize()
{
    _moveAnimation.setCenter(sf::Vector2f(90, 168));
    _moveTextureID = GameRender::registerTexture("data/textures/zombie/zombie_move.png");
    _attackTextureID = GameRender::registerTexture("data/textures/zombie/zombie_attack.png");

    _moveAnimation.setTextureID(_moveTextureID);
    _attackAnimation.setTextureID(_attackTextureID);
}

void Zombie::hit(WorldEntity* entity, GameWorld* gameWorld)
{
    switch(entity->getType())
    {
        case(EntityTypes::BULLET) :
        {
            Bullet* bullet = static_cast<Bullet*>(entity);
            const Vec2& pos(bullet->getCoord());
            float bulletAngle = bullet->getAngle();

            m_thisBody()->accelerate2D(bullet->getImpactForce());
            addLife(-bullet->getDamage());
            _time = getRandUnder(1000.0f);

            if (GameRender::isVisible(this))
            {
                gameWorld->addEntity(ExplosionProvider::getBase(pos));
                
                if (bullet->getDistance() < 50.0f)
                {
                    gameWorld->addEntity(ExplosionProvider::getClose(pos, bulletAngle));
                    gameWorld->addEntity(Guts::add(pos, bullet->getV()*40.f));
                }
            }

            break;
        }
        case(EntityTypes::HUNTER) :
        {
            if (_currentState != ATTACKING)
            {
                _currentState     = ATTACKING;
                _currentAnimation = _attackAnimation;
                _currentAnimation.resetTime(_time);
            }
            break;
        }
        default:
        {
            break;
        }
    }
}

void Zombie::initPhysics(GameWorld* world)
{
    m_initBody(world);
}