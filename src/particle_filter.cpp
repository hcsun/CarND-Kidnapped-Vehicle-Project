/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 50;

    normal_distribution<double> gps_x(0, std[0]);
    normal_distribution<double> gps_y(0, std[1]);
    normal_distribution<double> gps_h(0, std[2]);

    particles.resize(num_particles);
    for (int i = 0; i < particles.size(); i++) {
        particles[i].id = i;
        particles[i].x = x + gps_x(gen);
        particles[i].y = y + gps_y(gen);
        particles[i].theta = theta + gps_h(gen);
        particles[i].weight = 1.0f;
        weights.push_back(particles[i].weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
    normal_distribution<double> pos_x(0, std_pos[0]);
    normal_distribution<double> pos_y(0, std_pos[1]);
    normal_distribution<double> pos_h(0, std_pos[2]);

    for (int i = 0; i < particles.size(); i++) {
        if (fabs(yaw_rate) < 0.00001) {  
            particles[i].x += velocity * delta_t * cos(particles[i].theta);
            particles[i].y += velocity * delta_t * sin(particles[i].theta);
        } else { 
            double pred_x = particles[i].x + velocity / yaw_rate * (sin(particles[i].theta + delta_t * yaw_rate) - sin(particles[i].theta));
            double pred_y = particles[i].y + velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + delta_t * yaw_rate));
            double pred_theta = particles[i].theta + delta_t * yaw_rate;

            particles[i].x = pred_x + pos_x(gen);
            particles[i].y = pred_y + pos_y(gen);
            particles[i].theta = pred_theta + pos_h(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    for (int i = 0; i < observations.size(); i++) {    
        double min_dist = numeric_limits<double>::max();
        
        for (int j = 0; j < predicted.size(); j++) {          
            double distance = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

            if (distance < min_dist) {
                min_dist = distance;
                observations[i].id = predicted[j].id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
    for (int i = 0; i < particles.size(); i++) {
        //To keep landmarks only within sensor_range
        vector<LandmarkObs> pred_landmarks;
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            LandmarkObs candidate;

            candidate.x = map_landmarks.landmark_list[j].x_f;
            candidate.y = map_landmarks.landmark_list[j].y_f;
            candidate.id = map_landmarks.landmark_list[j].id_i;

            if (dist(particles[i].x, particles[i].y, candidate.x, candidate.y) < sensor_range) {
                pred_landmarks.push_back(candidate);
            }
        }

        //Transform local coordinate to world coordinate for observations
        vector<LandmarkObs> transformed_observation;
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs transformed;

            //rotation and translation
            transformed.x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
            transformed.y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;
            transformed.id = observations[j].id;

            transformed_observation.push_back(transformed);
        }

        dataAssociation(pred_landmarks, transformed_observation);


        double sigma_x_2 = 2 * pow(std_landmark[0], 2);
        double sigma_y_2 = 2 * pow(std_landmark[1], 2);
        double sigma_xy = 2 * M_PI * std_landmark[0] * std_landmark[1];

        //update weight
        particles[i].weight = 1.0;
        for (int j = 0; j < transformed_observation.size(); j++) {
            LandmarkObs observed;
            observed.x = transformed_observation[j].x;
            observed.y = transformed_observation[j].y;
            observed.id = transformed_observation[j].id;

            //Find associated landmark
            LandmarkObs target;
            for (int k = 0; k < pred_landmarks.size(); k++) {
                if (pred_landmarks[k].id == observed.id) {
                    target.x = pred_landmarks[k].x;
                    target.y = pred_landmarks[k].y;
                }
            }

            double diff_x_2 = pow(target.x - observed.x, 2);
            double diff_y_2 = pow(target.y - observed.y, 2);

            //calculate weight with multivariate Gaussian
            double weight = exp(-(diff_x_2 / sigma_x_2 + diff_y_2 / sigma_y_2)) / sigma_xy;

            particles[i].weight *= weight;
        }
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<int> select_index(weights.begin(), weights.end());
    
    vector<Particle> resamp_particles;
    
    for (int i = 0; i < num_particles; i++) {
        resamp_particles.push_back(particles[select_index(gen)]);
    }

    particles = resamp_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
