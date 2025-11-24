"""
IoT Integration Framework for Smart Waste Management
Enables communication with IoT-enabled smart bins and edge devices
Supports MQTT, WebSocket, and REST APIs for real-time monitoring
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BinStatus(str, Enum):
    """Smart bin operational status"""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    FULL = "full"
    ERROR = "error"


class WasteLevel(str, Enum):
    """Bin fill level indicators"""
    EMPTY = "empty"           # 0-25%
    LOW = "low"               # 25-50%
    MEDIUM = "medium"         # 50-75%
    HIGH = "high"             # 75-90%
    CRITICAL = "critical"     # 90-100%
    OVERFLOWING = "overflowing"  # >100%


class SmartBin:
    """
    Represents an IoT-enabled smart waste bin
    Tracks location, status, waste composition, and collection schedule
    """
    
    def __init__(self, bin_id: str, location: Dict, bin_type: str = "mixed"):
        self.bin_id = bin_id
        self.location = location  # {"lat": ..., "lon": ..., "address": ...}
        self.bin_type = bin_type  # "recyclable", "organic", "hazardous", "mixed"
        self.status = BinStatus.ONLINE
        self.fill_level_percent = 0.0
        self.waste_composition = {
            "recyclable": 0,
            "non_recyclable": 0,
            "healthy": 0,
            "hazardous": 0
        }
        self.last_updated = datetime.now()
        self.last_collection = None
        self.next_collection = None
        self.total_collections = 0
        self.sensors = {
            "ultrasonic_distance": None,  # cm
            "weight": None,                # kg
            "temperature": None,           # °C
            "gas_sensor": None,            # ppm (detect decomposition)
            "camera": True                 # Our detection system
        }
        self.alerts = []
    
    def update_fill_level(self, percent: float):
        """Update bin fill level and generate alerts if needed"""
        self.fill_level_percent = min(100.0, max(0.0, percent))
        self.last_updated = datetime.now()
        
        # Generate alerts based on fill level
        level = self.get_fill_level_category()
        
        if level == WasteLevel.CRITICAL:
            self.add_alert("CRITICAL", f"Bin {self.bin_id} is {percent:.0f}% full - collection needed soon")
        elif level == WasteLevel.OVERFLOWING:
            self.add_alert("EMERGENCY", f"Bin {self.bin_id} is overflowing - immediate collection required")
    
    def get_fill_level_category(self) -> WasteLevel:
        """Convert percentage to category"""
        if self.fill_level_percent <= 25:
            return WasteLevel.EMPTY
        elif self.fill_level_percent <= 50:
            return WasteLevel.LOW
        elif self.fill_level_percent <= 75:
            return WasteLevel.MEDIUM
        elif self.fill_level_percent <= 90:
            return WasteLevel.HIGH
        elif self.fill_level_percent <= 100:
            return WasteLevel.CRITICAL
        else:
            return WasteLevel.OVERFLOWING
    
    def add_waste_detection(self, category: str, count: int = 1):
        """Record waste detection in bin"""
        if category in self.waste_composition:
            self.waste_composition[category] += count
        
        # Check for hazardous waste
        if category == "hazardous":
            self.add_alert("HAZARD", f"Hazardous waste detected in bin {self.bin_id} - special handling required")
        
        # Estimate fill level increase (rough approximation)
        volume_per_item = 2.0  # 2% per item (adjustable)
        self.update_fill_level(self.fill_level_percent + (volume_per_item * count))
    
    def add_alert(self, severity: str, message: str):
        """Add alert to bin's alert queue"""
        alert = {
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "bin_id": self.bin_id,
            "location": self.location
        }
        self.alerts.append(alert)
        logger.warning(f"ALERT: {message}")
    
    def get_status(self) -> Dict:
        """Get comprehensive bin status"""
        return {
            "bin_id": self.bin_id,
            "location": self.location,
            "bin_type": self.bin_type,
            "status": self.status,
            "fill_level_percent": round(self.fill_level_percent, 1),
            "fill_level_category": self.get_fill_level_category(),
            "waste_composition": self.waste_composition,
            "last_updated": self.last_updated.isoformat(),
            "last_collection": self.last_collection.isoformat() if self.last_collection else None,
            "next_collection": self.next_collection.isoformat() if self.next_collection else None,
            "total_collections": self.total_collections,
            "sensors": self.sensors,
            "alerts": self.alerts,
            "collection_needed": self.fill_level_percent >= 85
        }
    
    def mark_collected(self):
        """Mark bin as collected and reset counters"""
        self.last_collection = datetime.now()
        self.fill_level_percent = 0.0
        self.waste_composition = {k: 0 for k in self.waste_composition}
        self.alerts = []
        self.total_collections += 1
        logger.info(f"Bin {self.bin_id} marked as collected (total: {self.total_collections})")


class IoTManager:
    """
    Central IoT management system
    Coordinates multiple smart bins, handles MQTT/WebSocket, generates analytics
    """
    
    def __init__(self):
        self.bins: Dict[str, SmartBin] = {}
        self.mqtt_client = None  # Placeholder for MQTT client
        self.websocket_connections = []  # Active WebSocket connections
        self.event_callbacks: List[Callable] = []
    
    def register_bin(self, bin_id: str, location: Dict, bin_type: str = "mixed") -> SmartBin:
        """Register a new smart bin in the system"""
        if bin_id in self.bins:
            logger.warning(f"Bin {bin_id} already registered - updating")
        
        bin_obj = SmartBin(bin_id, location, bin_type)
        self.bins[bin_id] = bin_obj
        logger.info(f"Registered bin {bin_id} at {location.get('address', 'unknown location')}")
        
        return bin_obj
    
    def get_bin(self, bin_id: str) -> Optional[SmartBin]:
        """Get bin by ID"""
        return self.bins.get(bin_id)
    
    def update_bin_from_detection(self, bin_id: str, detection_results: Dict):
        """
        Update bin status from waste detection results
        
        Args:
            bin_id: Smart bin identifier
            detection_results: Results from waste categorizer
        """
        bin_obj = self.get_bin(bin_id)
        if not bin_obj:
            logger.warning(f"Bin {bin_id} not found - skipping update")
            return
        
        # Update waste composition
        category_breakdown = detection_results.get('category_breakdown', {})
        for category, count in category_breakdown.items():
            bin_obj.add_waste_detection(category, count)
        
        # Check for hazardous alerts
        if detection_results.get('hazardous_alert', False):
            hazard_items = detection_results.get('hazardous_items', [])
            for item in hazard_items:
                bin_obj.add_alert("HAZARD", f"Hazardous: {item['class']} - {item['instructions']}")
        
        # Trigger event callbacks
        self._trigger_event("bin_updated", bin_obj.get_status())
    
    def get_collection_route(self, priority: str = "critical") -> List[Dict]:
        """
        Generate optimized collection route for waste trucks
        
        Args:
            priority: Filter by fill level ("critical", "high", "all")
            
        Returns:
            List of bins sorted by priority and proximity
        """
        # Filter bins by priority
        if priority == "critical":
            target_bins = [b for b in self.bins.values() if b.fill_level_percent >= 90]
        elif priority == "high":
            target_bins = [b for b in self.bins.values() if b.fill_level_percent >= 75]
        else:
            target_bins = list(self.bins.values())
        
        # Sort by fill level (descending)
        target_bins.sort(key=lambda b: b.fill_level_percent, reverse=True)
        
        # NOTE: Future enhancement - implement TSP (Traveling Salesman Problem) algorithm
        # with actual GPS coordinates for optimal route planning
        # Current implementation prioritizes bins by fill level
        
        return [
            {
                "bin_id": b.bin_id,
                "location": b.location,
                "fill_level": round(b.fill_level_percent, 1),
                "priority": "CRITICAL" if b.fill_level_percent >= 90 else "HIGH" if b.fill_level_percent >= 75 else "MEDIUM",
                "waste_composition": b.waste_composition,
                "hazardous_present": b.waste_composition.get("hazardous", 0) > 0
            }
            for b in target_bins
        ]
    
    def get_system_dashboard(self) -> Dict:
        """Generate comprehensive dashboard data"""
        total_bins = len(self.bins)
        if total_bins == 0:
            return {"error": "No bins registered"}
        
        # Aggregate statistics
        total_waste = sum(sum(b.waste_composition.values()) for b in self.bins.values())
        avg_fill = sum(b.fill_level_percent for b in self.bins.values()) / total_bins
        
        bins_by_status = {
            "online": sum(1 for b in self.bins.values() if b.status == BinStatus.ONLINE),
            "offline": sum(1 for b in self.bins.values() if b.status == BinStatus.OFFLINE),
            "full": sum(1 for b in self.bins.values() if b.fill_level_percent >= 90),
            "needs_collection": sum(1 for b in self.bins.values() if b.fill_level_percent >= 85)
        }
        
        # Waste composition aggregation
        total_composition = {
            "recyclable": sum(b.waste_composition.get("recyclable", 0) for b in self.bins.values()),
            "non_recyclable": sum(b.waste_composition.get("non_recyclable", 0) for b in self.bins.values()),
            "healthy": sum(b.waste_composition.get("healthy", 0) for b in self.bins.values()),
            "hazardous": sum(b.waste_composition.get("hazardous", 0) for b in self.bins.values())
        }
        
        # Active alerts
        all_alerts = []
        for bin_obj in self.bins.values():
            all_alerts.extend(bin_obj.alerts)
        all_alerts.sort(key=lambda a: a["timestamp"], reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_bins": total_bins,
            "bins_by_status": bins_by_status,
            "average_fill_level": round(avg_fill, 1),
            "total_waste_items": total_waste,
            "waste_composition": total_composition,
            "recycling_rate": round((total_composition["recyclable"] / max(total_waste, 1)) * 100, 1),
            "active_alerts": len(all_alerts),
            "critical_alerts": sum(1 for a in all_alerts if a["severity"] in ["CRITICAL", "EMERGENCY", "HAZARD"]),
            "alerts": all_alerts[:10],  # Latest 10 alerts
            "collection_queue": len([b for b in self.bins.values() if b.fill_level_percent >= 85]),
            "bins_needing_attention": [
                {
                    "bin_id": b.bin_id,
                    "issue": "Full" if b.fill_level_percent >= 90 else "Hazardous" if b.waste_composition.get("hazardous", 0) > 0 else "Offline",
                    "location": b.location,
                    "fill_level": round(b.fill_level_percent, 1)
                }
                for b in self.bins.values()
                if b.fill_level_percent >= 85 or b.waste_composition.get("hazardous", 0) > 0 or b.status != BinStatus.ONLINE
            ]
        }
    
    def register_event_callback(self, callback: Callable):
        """Register callback for IoT events"""
        self.event_callbacks.append(callback)
    
    def _trigger_event(self, event_type: str, data: Dict):
        """Trigger registered event callbacks"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    async def simulate_mqtt_publish(self, topic: str, payload: Dict):
        """
        Simulate MQTT publish (replace with actual MQTT client)
        
        Example topics:
        - waste/bins/{bin_id}/status
        - waste/bins/{bin_id}/detection
        - waste/alerts
        """
        logger.info(f"MQTT Publish: {topic}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
        
        # NOTE: For production deployment, integrate paho-mqtt client:
        # Install: pip install paho-mqtt
        # Uncomment: self.mqtt_client.publish(topic, json.dumps(payload))
    
    async def handle_mqtt_message(self, topic: str, payload: str):
        """
        Handle incoming MQTT messages from smart bins
        
        Example message from bin sensor:
        {
            "bin_id": "BIN-001",
            "distance_cm": 45,
            "weight_kg": 12.5,
            "temperature_c": 22
        }
        """
        try:
            data = json.loads(payload)
            bin_id = data.get("bin_id")
            
            if not bin_id:
                logger.warning("Received MQTT message without bin_id")
                return
            
            bin_obj = self.get_bin(bin_id)
            if not bin_obj:
                logger.warning(f"Unknown bin {bin_id} - ignoring message")
                return
            
            # Update sensor data
            if "distance_cm" in data:
                bin_obj.sensors["ultrasonic_distance"] = data["distance_cm"]
                # Convert distance to fill level (bin height - distance) / bin height
                bin_height_cm = 100  # Configurable
                fill_percent = ((bin_height_cm - data["distance_cm"]) / bin_height_cm) * 100
                bin_obj.update_fill_level(fill_percent)
            
            if "weight_kg" in data:
                bin_obj.sensors["weight"] = data["weight_kg"]
            
            if "temperature_c" in data:
                bin_obj.sensors["temperature"] = data["temperature_c"]
            
            logger.info(f"Updated bin {bin_id} from MQTT message")
            
        except Exception as e:
            logger.error(f"Error handling MQTT message: {e}")


# Singleton instance
_iot_manager_instance = None

def get_iot_manager() -> IoTManager:
    """Get singleton instance of IoT manager"""
    global _iot_manager_instance
    if _iot_manager_instance is None:
        _iot_manager_instance = IoTManager()
        logger.info("IoT Manager initialized")
        
        # Register demo bins for testing
        _iot_manager_instance.register_bin(
            "BIN-001",
            {"lat": 40.7128, "lon": -74.0060, "address": "123 Main St, New York, NY"},
            "mixed"
        )
        _iot_manager_instance.register_bin(
            "BIN-002",
            {"lat": 40.7589, "lon": -73.9851, "address": "456 Park Ave, New York, NY"},
            "recyclable"
        )
        _iot_manager_instance.register_bin(
            "BIN-003",
            {"lat": 40.7484, "lon": -73.9857, "address": "789 5th Ave, New York, NY"},
            "organic"
        )
        
    return _iot_manager_instance


# MQTT Protocol Example
"""
To integrate with actual MQTT broker (e.g., Mosquitto, AWS IoT):

1. Install: pip install paho-mqtt

2. Setup MQTT client:
    import paho.mqtt.client as mqtt
    
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect("mqtt.broker.com", 1883, 60)
    client.loop_start()

3. Subscribe to bin topics:
    client.subscribe("waste/bins/+/sensors")
    client.subscribe("waste/bins/+/status")

4. Publish detection results:
    manager = get_iot_manager()
    client.publish(
        f"waste/bins/{bin_id}/detection",
        json.dumps(detection_results)
    )
"""

# WebSocket Protocol Example
"""
For real-time dashboard updates:

1. Install: pip install websockets

2. Setup WebSocket server:
    async def handler(websocket, path):
        manager = get_iot_manager()
        
        while True:
            dashboard = manager.get_system_dashboard()
            await websocket.send(json.dumps(dashboard))
            await asyncio.sleep(5)  # Update every 5 seconds

3. Client connects:
    ws://your-server.com:8765/dashboard
"""
