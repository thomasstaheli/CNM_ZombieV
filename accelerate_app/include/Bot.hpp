#ifndef BOT_HPP_INCLUDED
#define BOT_HPP_INCLUDED

#include "HunterBase.hpp"

class Bot : public HunterBase, public WorldEntityPool<Bot>
{
public:
    Bot();
    Bot(float x, float y);

    void hit(WorldEntity* entity, GameWorld* gameWorld);
    void setTarget(WorldEntity* entity) {m_target = entity;}
    void update(GameWorld& world);

    static void initialize();

private:
    WorldEntity* m_target;
    size_t m_getTargetCount;
    
    // Optimisations
    float m_targetSearchCooldown;  // Cooldown avant de rechercher un nouveau target
    float m_lastBotX, m_lastBotY;  // Cache de la dernière position pour éviter getCoord()
    
    static const float TARGET_SEARCH_INTERVAL;  // 0.5 secondes entre les recherches

    void computeControls(GameWorld& world);
    void getTarget(GameWorld* world);
};

#endif // BOT_HPP_INCLUDED