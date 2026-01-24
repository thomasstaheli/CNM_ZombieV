#include <iostream>
#include <array>
#include <SFML/Graphics.hpp>

#include <fstream> // Added for profiling
#include <deque>

#include "System/GameWorld.hpp"
#include "System/GameRender.hpp"

#include "Bot.hpp"
#include "Props/Props.hpp"
#include "Turret.hpp"
#include "Hunter.hpp"
#include "Zombie.hpp"

#define WIN_WIDTH 1600
#define WIN_HEIGHT 900

#include "System/Utils.hpp"

int main()
{
    sf::ContextSettings settings;
    settings.antialiasingLevel = 0;
    sf::RenderWindow window(sf::VideoMode(WIN_WIDTH, WIN_HEIGHT), "Zombie V", sf::Style::Default, settings);
    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    GameRender::initialize(WIN_WIDTH, WIN_HEIGHT);
    GameWorld world;
    world.initEventHandler(window);

    Hunter::registerObject(&world);
    Bot::registerObject(&world);

	world.initializeWeapons();

    world.getPhyManager().setGravity(Vec2(0, 0));

    Hunter& h = *Hunter::newEntity(static_cast<float>(MAP_SIZE/2), static_cast<float>(MAP_SIZE/2));
    // Tableau de bot
    std::array<Bot*, 10> bots;
    // Joueur
    world.addEntity(&h);

    int waveCount = 0;

    // Création de 10 bots
    
    for (int i = 0; i < 10; ++i)
    {
        bots[i] = Bot::newEntity(static_cast<float>(MAP_SIZE / 2 + rand() % 10), 
                                static_cast<float>(MAP_SIZE / 2 + rand() % 10));
        world.addEntity(bots[i]);
    }

    sf::Mouse::setPosition(sf::Vector2i(WIN_WIDTH/2+100, WIN_HEIGHT/2));

    Zombie* newZombie;

    // Nombre de zombies initiaux à la vague numéro 1
    for (int i(100); i--;)
    {
        newZombie = Zombie::newEntity(getRandUnder(static_cast<float>(MAP_SIZE)), getRandUnder(static_cast<float>(MAP_SIZE)));
		Bot& bot = *bots[i % 10];
		EntityID target; // = bot.getID();
        target = h.getID();
		newZombie->setTarget(target);
        world.addEntity(newZombie);
    }

    for (int i(0); i<10; ++i)
    {
        Light light;
        light.position = Vec2(getRandUnder(2000.0f), getRandUnder(2000.0f));
        light.color    = sf::Color(rand()%255, rand()%255,rand()%255);
        light.radius   = getRandFloat(300.0f, 450.0f);
        GameRender::getLightEngine().addDurableLight(light);
    }

    // <--- AJOUT : Préparation du fichier CSV
    std::ofstream csvFile("mesures_fps.csv");
    // On écrit l'en-tête des colonnes (Le point-virgule est souvent mieux reconnu par Excel qu'une virgule)
    csvFile << "Vague;Frame_Index;Temps_Total(ms);FPS_Instantane\n";
    
    int frameCountGlobal = 0;
    sf::Clock frameClock; // Clock for profiling
    int frames = 0;
    float timer = 0;
    std::deque<float> last10Frames; // File pour garder les 10 dernières mesures

    while (window.isOpen())
    {
        sf::Time dt = frameClock.restart();
        float dtMillis = dt.asMilliseconds();

        // 2. Gestion du stockage des 10 dernières mesures
        last10Frames.push_back(dtMillis);
        if (last10Frames.size() > 10) {
            last10Frames.pop_front(); // On enlève la plus vieille pour garder toujours 10 éléments
        }

        // 3. Mise à jour des compteurs "seconde"
        timer += dt.asSeconds();
        frames++;

        // Logique d'enregistrement CSV
        if (timer >= 1.0f) 
        {
            float avgFps = frames / timer; // Calcul précis
            
            // A. Mise à jour du titre
            std::string title = "Zombie V - Wave: " + std::to_string(waveCount) + 
                                " - FPS: " + std::to_string((int)avgFps) + 
                                " (" + std::to_string(1000.0f/avgFps) + " ms)";
            window.setTitle(title);

            // B. Écriture dans le CSV
            // On écrit : Vague ; FPS Moyen ; F1 ; F2 ... ; F10
            csvFile << waveCount << ";" << (int)avgFps;
            
            for (float t : last10Frames) {
                csvFile << ";" << t;
            }
            // Si jamais on a moins de 10 frames (ex: début du programme ou lag extrême), on comble avec des vides
            for (size_t i = last10Frames.size(); i < 10; ++i) {
                csvFile << ";"; 
            }
            csvFile << "\n";

            // C. Reset pour la seconde suivante
            frames = 0;
            timer = 0;
        }

        ++frameCountGlobal;
        if (Zombie::getObjectsCount() == 0)
        {
            ++waveCount;
            // for (int i(waveCount*waveCount + 10); i--;)
            for (int i(waveCount*1000 + 100); i--;)
            {
                Zombie* newZombie(Zombie::newEntity(getRandUnder(static_cast<float>(MAP_SIZE)), getRandUnder(static_cast<float>(MAP_SIZE))));
                //newZombie->setTarget(&(*Hunter::getObjects().front()));
                world.addEntity(newZombie);
            }
        }

        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed) window.close();
            if (event.type == sf::Event::KeyPressed)
                if (event.key.code == sf::Keyboard::Escape) window.close();
        }

        sf::Clock clock;
        // Update de toutes les entités dans le monde
        world.update();

        Vec2 p = h.getCoord();
		GameRender::setFocus({ p.x, p.y });

        GameRender::clear();

        world.render();
        GameRender::display(&window);

        window.display();
    }

    // Fermeture propre du fichier
    csvFile.close();

    return 0;
}
