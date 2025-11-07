"""
Enhanced Geo-Verification System with Configurable Radius and Advanced Location Services
"""

import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeoLocation:
    """Represents a geographical location with accuracy metrics"""
    latitude: float
    longitude: float
    accuracy: float  # GPS accuracy in meters
    altitude: Optional[float] = None
    timestamp: Optional[str] = None

@dataclass
class GeoFenceZone:
    """Represents a geo-fenced area with configurable parameters"""
    name: str
    center_lat: float
    center_lng: float
    radius_meters: float
    polygon_coordinates: List[Dict[str, float]]
    buffer_zone: float = 10.0  # Additional buffer in meters
    is_active: bool = True
    created_by: str = "admin"
    created_at: str = None

class AdvancedGeoVerification:
    """
    Advanced geo-verification system with multiple validation methods
    """
    
    def __init__(self, default_radius: float = 50.0):
        self.default_radius = default_radius
        self.earth_radius = 6371000  # Earth's radius in meters
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate distance between two points using Haversine formula
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        return self.earth_radius * c
    
    def verify_circular_geofence(self, user_location: GeoLocation, geofence: GeoFenceZone) -> Dict[str, any]:
        """
        Verify if user is within circular geo-fence with configurable radius
        """
        try:
            distance = self.calculate_distance(
                user_location.latitude, user_location.longitude,
                geofence.center_lat, geofence.center_lng
            )
            
            # Add buffer zone to account for GPS accuracy
            effective_radius = geofence.radius_meters + geofence.buffer_zone
            
            is_within_radius = distance <= effective_radius
            
            return {
                'is_within_geofence': is_within_radius,
                'distance_from_center': distance,
                'allowed_radius': effective_radius,
                'gps_accuracy': user_location.accuracy,
                'confidence_level': self._calculate_confidence(user_location, distance, effective_radius)
            }
            
        except Exception as e:
            logger.error(f"Error in circular geofence verification: {e}")
            return {
                'is_within_geofence': False,
                'error': str(e)
            }
    
    def verify_polygon_geofence(self, user_location: GeoLocation, geofence: GeoFenceZone) -> Dict[str, any]:
        """
        Verify if user is within polygon geo-fence using ray casting algorithm
        """
        try:
            is_inside = self._point_in_polygon(
                user_location.latitude, user_location.longitude,
                geofence.polygon_coordinates
            )
            
            # Calculate distance to nearest polygon edge
            nearest_distance = self._distance_to_polygon_edge(
                user_location.latitude, user_location.longitude,
                geofence.polygon_coordinates
            )
            
            return {
                'is_within_geofence': is_inside,
                'distance_to_edge': nearest_distance,
                'gps_accuracy': user_location.accuracy,
                'confidence_level': self._calculate_polygon_confidence(user_location, is_inside, nearest_distance)
            }
            
        except Exception as e:
            logger.error(f"Error in polygon geofence verification: {e}")
            return {
                'is_within_geofence': False,
                'error': str(e)
            }
    
    def _point_in_polygon(self, lat: float, lng: float, polygon: List[Dict[str, float]]) -> bool:
        """
        Ray casting algorithm for point-in-polygon test
        """
        if not polygon or len(polygon) < 3:
            return False
        
        x, y = lat, lng
        inside = False
        
        j = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = polygon[i]['lat'], polygon[i]['lng']
            xj, yj = polygon[j]['lat'], polygon[j]['lng']
            
            if (((yi > y) != (yj > y)) and 
                (x < (xj - xi) * (y - yi) / (yj - yi) + xi)):
                inside = not inside
            j = i
        
        return inside
    
    def _distance_to_polygon_edge(self, lat: float, lng: float, polygon: List[Dict[str, float]]) -> float:
        """
        Calculate minimum distance to polygon edge
        """
        min_distance = float('inf')
        
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            # Calculate distance to line segment
            distance = self._point_to_line_distance(lat, lng, p1, p2)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _point_to_line_distance(self, lat: float, lng: float, p1: Dict[str, float], p2: Dict[str, float]) -> float:
        """
        Calculate distance from point to line segment
        """
        # Convert to meters for calculation
        x0, y0 = lat, lng
        x1, y1 = p1['lat'], p1['lng']
        x2, y2 = p2['lat'], p2['lng']
        
        # Calculate distance using perpendicular distance formula
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2
        
        distance = abs(A * x0 + B * y0 + C) / math.sqrt(A**2 + B**2)
        
        # Convert back to meters
        return distance * 111000  # Approximate conversion
    
    def _calculate_confidence(self, user_location: GeoLocation, distance: float, allowed_radius: float) -> float:
        """
        Calculate confidence level for geo-verification
        """
        # Base confidence on GPS accuracy and distance
        accuracy_factor = min(1.0, user_location.accuracy / 10.0)  # Better accuracy = higher confidence
        distance_factor = max(0.0, 1.0 - (distance / allowed_radius))  # Closer to center = higher confidence
        
        return (accuracy_factor + distance_factor) / 2.0
    
    def _calculate_polygon_confidence(self, user_location: GeoLocation, is_inside: bool, distance_to_edge: float) -> float:
        """
        Calculate confidence level for polygon geo-verification
        """
        accuracy_factor = min(1.0, user_location.accuracy / 10.0)
        
        if is_inside:
            # Higher confidence if well inside the polygon
            distance_factor = min(1.0, distance_to_edge / 50.0)  # 50m buffer
        else:
            # Lower confidence if outside
            distance_factor = max(0.0, 1.0 - (distance_to_edge / 100.0))
        
        return (accuracy_factor + distance_factor) / 2.0
    
    def create_geofence_zone(self, name: str, center_lat: float, center_lng: float, 
                           radius_meters: float, polygon_coords: List[Dict[str, float]] = None) -> GeoFenceZone:
        """
        Create a new geo-fence zone
        """
        if polygon_coords is None:
            # Create circular polygon from center and radius
            polygon_coords = self._create_circular_polygon(center_lat, center_lng, radius_meters)
        
        return GeoFenceZone(
            name=name,
            center_lat=center_lat,
            center_lng=center_lng,
            radius_meters=radius_meters,
            polygon_coordinates=polygon_coords
        )
    
    def _create_circular_polygon(self, center_lat: float, center_lng: float, radius_meters: float, 
                                num_points: int = 16) -> List[Dict[str, float]]:
        """
        Create a circular polygon from center point and radius
        """
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            # Convert radius to degrees (approximate)
            lat_offset = (radius_meters / 111000) * math.cos(angle)
            lng_offset = (radius_meters / (111000 * math.cos(math.radians(center_lat)))) * math.sin(angle)
            
            points.append({
                'lat': center_lat + lat_offset,
                'lng': center_lng + lng_offset
            })
        
        return points

class GeoVerificationManager:
    """
    Manager class for handling geo-verification operations
    """
    
    def __init__(self):
        self.geo_verifier = AdvancedGeoVerification()
    
    def verify_user_location(self, user_location: GeoLocation, geofence: GeoFenceZone) -> Dict[str, any]:
        """
        Comprehensive location verification
        """
        # Check GPS accuracy
        if user_location.accuracy > 50:  # More than 50m accuracy is unreliable
            return {
                'verification_passed': False,
                'error': 'GPS accuracy too low',
                'accuracy': user_location.accuracy
            }
        
        # Perform both circular and polygon verification
        circular_result = self.geo_verifier.verify_circular_geofence(user_location, geofence)
        polygon_result = self.geo_verifier.verify_polygon_geofence(user_location, geofence)
        
        # Combine results
        verification_passed = (circular_result.get('is_within_geofence', False) and 
                             polygon_result.get('is_within_geofence', False))
        
        return {
            'verification_passed': verification_passed,
            'circular_verification': circular_result,
            'polygon_verification': polygon_result,
            'user_location': {
                'latitude': user_location.latitude,
                'longitude': user_location.longitude,
                'accuracy': user_location.accuracy
            },
            'geofence_info': {
                'name': geofence.name,
                'radius': geofence.radius_meters,
                'center': [geofence.center_lat, geofence.center_lng]
            }
        }
