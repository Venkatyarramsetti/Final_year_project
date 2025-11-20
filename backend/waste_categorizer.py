"""
Advanced Waste Categorization System
Implements 4-category classification as per project abstract:
- Recyclable
- Non-Recyclable  
- Healthy (organic/food)
- Hazardous

Provides material analysis, environmental impact scoring, and recycling recommendations
"""

import logging
from typing import Dict, List, Tuple
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WasteCategory(str, Enum):
    """Main waste categories as per abstract"""
    RECYCLABLE = "recyclable"
    NON_RECYCLABLE = "non_recyclable"
    HEALTHY = "healthy"
    HAZARDOUS = "hazardous"


class MaterialType(str, Enum):
    """Material composition types"""
    PLASTIC = "plastic"
    GLASS = "glass"
    METAL = "metal"
    PAPER = "paper"
    ORGANIC = "organic"
    MIXED = "mixed"
    CHEMICAL = "chemical"
    ELECTRONIC = "electronic"


class RecyclingGrade(str, Enum):
    """Recycling difficulty grades"""
    EASY = "easy"           # Clean, single-material
    MODERATE = "moderate"   # Requires processing
    DIFFICULT = "difficult" # Complex recycling
    NOT_RECYCLABLE = "not_recyclable"


class WasteCategorizer:
    """
    Intelligent waste categorization engine
    Maps YOLO detections to waste categories and provides actionable insights
    """
    
    def __init__(self):
        # Recyclable materials database
        self.recyclable_items = {
            # Plastics (check recycling symbols)
            'plastic_bottle': {
                'material': MaterialType.PLASTIC,
                'grade': RecyclingGrade.EASY,
                'instructions': 'Rinse clean, remove cap, flatten',
                'impact': 'high',  # High environmental impact if not recycled
                'recycling_symbol': 'PET (1)',
                'processing_facility': 'municipal_recycling'
            },
            'plastic_container': {
                'material': MaterialType.PLASTIC,
                'grade': RecyclingGrade.MODERATE,
                'instructions': 'Remove food residue, check symbol',
                'impact': 'high',
                'recycling_symbol': 'HDPE (2) or PP (5)',
                'processing_facility': 'municipal_recycling'
            },
            'plastic_bag': {
                'material': MaterialType.PLASTIC,
                'grade': RecyclingGrade.DIFFICULT,
                'instructions': 'Return to store collection bins',
                'impact': 'very_high',  # Often ends in ocean
                'recycling_symbol': 'LDPE (4)',
                'processing_facility': 'specialized_facility'
            },
            
            # Glass
            'glass_bottle': {
                'material': MaterialType.GLASS,
                'grade': RecyclingGrade.EASY,
                'instructions': 'Rinse, remove caps, separate by color',
                'impact': 'low',  # Glass is highly recyclable
                'recycling_symbol': 'GL (70-74)',
                'processing_facility': 'municipal_recycling'
            },
            
            # Metals
            'metal_can': {
                'material': MaterialType.METAL,
                'grade': RecyclingGrade.EASY,
                'instructions': 'Rinse clean, can crush',
                'impact': 'low',  # Metals recycle well
                'recycling_symbol': 'ALU (41) or FE (40)',
                'processing_facility': 'municipal_recycling'
            },
            
            # Paper products
            'paper': {
                'material': MaterialType.PAPER,
                'grade': RecyclingGrade.EASY,
                'instructions': 'Keep dry and clean, no food stains',
                'impact': 'low',
                'recycling_symbol': 'PAP (20-22)',
                'processing_facility': 'municipal_recycling'
            },
            'cardboard': {
                'material': MaterialType.PAPER,
                'grade': RecyclingGrade.EASY,
                'instructions': 'Flatten boxes, remove tape/staples',
                'impact': 'low',
                'recycling_symbol': 'PAP (20)',
                'processing_facility': 'municipal_recycling'
            }
        }
        
        # Non-recyclable items (contaminated or mixed materials)
        self.non_recyclable_items = {
            'wrapper': {
                'material': MaterialType.MIXED,
                'reason': 'Mixed materials (plastic + foil)',
                'disposal': 'landfill',
                'impact': 'high'
            },
            'styrofoam': {
                'material': MaterialType.PLASTIC,
                'reason': 'Not accepted by most facilities',
                'disposal': 'landfill',
                'impact': 'very_high'
            },
            'contaminated_container': {
                'material': MaterialType.MIXED,
                'reason': 'Food contamination',
                'disposal': 'landfill or compost if organic',
                'impact': 'moderate'
            },
            'general_trash': {
                'material': MaterialType.MIXED,
                'reason': 'Mixed waste',
                'disposal': 'landfill',
                'impact': 'moderate'
            }
        }
        
        # Healthy items (fresh food, organic materials)
        self.healthy_items = {
            'banana': {'type': 'fresh_fruit', 'compostable': True},
            'apple': {'type': 'fresh_fruit', 'compostable': True},
            'orange': {'type': 'fresh_fruit', 'compostable': True},
            'carrot': {'type': 'fresh_vegetable', 'compostable': True},
            'broccoli': {'type': 'fresh_vegetable', 'compostable': True},
            'lettuce': {'type': 'fresh_vegetable', 'compostable': True},
            'food_waste': {
                'type': 'organic_waste',
                'compostable': True,
                'instructions': 'Separate from packaging, compost',
                'disposal': 'composting_facility',
                'environmental_benefit': 'Reduces methane in landfills'
            }
        }
        
        # Hazardous materials (requires special handling)
        self.hazardous_items = {
            'battery': {
                'hazard_type': 'chemical',
                'severity': 'high',
                'risks': ['Heavy metals', 'Acid leakage', 'Fire hazard'],
                'disposal': 'hazardous_waste_facility',
                'instructions': 'Never dispose in regular trash',
                'regulations': 'EPA regulated',
                'environmental_impact': 'Soil and water contamination'
            },
            'syringe': {
                'hazard_type': 'medical',
                'severity': 'critical',
                'risks': ['Bloodborne pathogens', 'Needle stick injury'],
                'disposal': 'sharps_container → medical_waste',
                'instructions': 'Use sharps container immediately',
                'regulations': 'OSHA regulated'
            },
            'chemical_container': {
                'hazard_type': 'chemical',
                'severity': 'high',
                'risks': ['Toxic fumes', 'Skin burns', 'Environmental contamination'],
                'disposal': 'hazardous_waste_facility',
                'instructions': 'Keep container sealed, check SDS',
                'regulations': 'EPA regulated'
            },
            'broken_glass': {
                'hazard_type': 'physical',
                'severity': 'medium',
                'risks': ['Cuts', 'Lacerations'],
                'disposal': 'Wrap securely, label as "BROKEN GLASS"',
                'instructions': 'Do not recycle with regular glass'
            },
            'electronic_waste': {
                'hazard_type': 'electronic',
                'severity': 'medium',
                'risks': ['Heavy metals', 'Toxic components'],
                'disposal': 'e_waste_recycling_facility',
                'instructions': 'Remove batteries, erase data',
                'regulations': 'WEEE Directive (EU) / EPA (US)',
                'recycling_value': 'Contains valuable metals (gold, copper)'
            }
        }
    
    def categorize(self, class_name: str, confidence: float, context: dict = None) -> Dict:
        """
        Categorize detected object into 4 main waste categories
        
        Args:
            class_name: Detected class from YOLO
            confidence: Detection confidence
            context: Additional context (in_bin, location, etc.)
            
        Returns:
            Dictionary with category, material, handling instructions, and impact
        """
        class_lower = class_name.lower()
        
        # Check each category in priority order
        
        # 1. Hazardous (highest priority - safety critical)
        if class_lower in self.hazardous_items:
            return self._create_hazardous_response(class_lower, confidence)
        
        # 2. Healthy (fresh food, no processing needed)
        if class_lower in self.healthy_items:
            return self._create_healthy_response(class_lower, confidence)
        
        # 3. Recyclable (materials that can be recycled)
        if class_lower in self.recyclable_items:
            return self._create_recyclable_response(class_lower, confidence, context)
        
        # 4. Non-recyclable (default for mixed/contaminated waste)
        if class_lower in self.non_recyclable_items:
            return self._create_non_recyclable_response(class_lower, confidence)
        
        # Default: Try to infer from class name
        return self._infer_category(class_lower, confidence, context)
    
    def _create_hazardous_response(self, class_name: str, confidence: float) -> Dict:
        """Create response for hazardous materials"""
        item_data = self.hazardous_items[class_name]
        
        return {
            'category': WasteCategory.HAZARDOUS,
            'class': class_name,
            'confidence': confidence,
            'hazard_type': item_data['hazard_type'],
            'severity': item_data['severity'],
            'risks': item_data['risks'],
            'disposal_method': item_data['disposal'],
            'instructions': item_data['instructions'],
            'regulations': item_data.get('regulations', 'Check local regulations'),
            'environmental_impact': item_data.get('environmental_impact', 'High risk if improperly disposed'),
            'human_exposure_risk': 'HIGH - Avoid direct contact',
            'requires_special_handling': True,
            'color_code': 'red',  # For UI display
            'icon': '⚠️'
        }
    
    def _create_healthy_response(self, class_name: str, confidence: float) -> Dict:
        """Create response for healthy/fresh items"""
        item_data = self.healthy_items[class_name]
        
        return {
            'category': WasteCategory.HEALTHY,
            'class': class_name,
            'confidence': confidence,
            'item_type': item_data['type'],
            'compostable': item_data['compostable'],
            'disposal_method': item_data.get('disposal', 'compost or consume'),
            'instructions': item_data.get('instructions', 'Fresh food - no processing needed'),
            'environmental_benefit': item_data.get('environmental_benefit', 'Organic material - compost to enrich soil'),
            'human_exposure_risk': 'NONE - Safe to handle',
            'requires_special_handling': False,
            'color_code': 'green',
            'icon': '✓'
        }
    
    def _create_recyclable_response(self, class_name: str, confidence: float, context: dict = None) -> Dict:
        """Create response for recyclable materials"""
        item_data = self.recyclable_items[class_name]
        
        # Check if item is contaminated (in trash bin)
        in_bin = context and context.get('in_bin', False)
        contaminated = in_bin or (context and context.get('contaminated', False))
        
        if contaminated:
            return {
                'category': WasteCategory.NON_RECYCLABLE,
                'class': class_name,
                'confidence': confidence,
                'original_material': item_data['material'],
                'reason': 'Contaminated or mixed with trash',
                'disposal_method': 'landfill (cannot recycle)',
                'instructions': 'Contaminated items cannot be recycled',
                'environmental_impact': 'Could have been recycled if clean',
                'color_code': 'orange',
                'icon': '⊘'
            }
        
        return {
            'category': WasteCategory.RECYCLABLE,
            'class': class_name,
            'confidence': confidence,
            'material': item_data['material'],
            'recycling_grade': item_data['grade'],
            'recycling_symbol': item_data['recycling_symbol'],
            'instructions': item_data['instructions'],
            'processing_facility': item_data['processing_facility'],
            'environmental_impact': item_data['impact'],
            'environmental_benefit': f'Recycling saves energy and reduces {item_data["material"]} production',
            'human_exposure_risk': 'LOW - Safe when clean',
            'requires_special_handling': False,
            'color_code': 'blue',
            'icon': '♻️'
        }
    
    def _create_non_recyclable_response(self, class_name: str, confidence: float) -> Dict:
        """Create response for non-recyclable waste"""
        item_data = self.non_recyclable_items.get(class_name, {})
        
        return {
            'category': WasteCategory.NON_RECYCLABLE,
            'class': class_name,
            'confidence': confidence,
            'material': item_data.get('material', MaterialType.MIXED),
            'reason': item_data.get('reason', 'Mixed materials or contamination'),
            'disposal_method': item_data.get('disposal', 'landfill'),
            'environmental_impact': item_data.get('impact', 'moderate'),
            'instructions': 'Cannot be recycled - dispose in general waste',
            'human_exposure_risk': 'LOW',
            'requires_special_handling': False,
            'color_code': 'gray',
            'icon': '🗑️'
        }
    
    def _infer_category(self, class_name: str, confidence: float, context: dict = None) -> Dict:
        """Infer category from class name keywords"""
        
        # Hazardous keywords
        if any(keyword in class_name for keyword in ['hazard', 'chemical', 'battery', 'medical', 'sharp', 'broken']):
            return {
                'category': WasteCategory.HAZARDOUS,
                'class': class_name,
                'confidence': confidence,
                'inferred': True,
                'severity': 'medium',
                'instructions': 'Treat as hazardous - special disposal required',
                'color_code': 'red',
                'icon': '⚠️'
            }
        
        # Recyclable keywords
        if any(keyword in class_name for keyword in ['bottle', 'can', 'glass', 'metal', 'paper', 'cardboard']):
            return {
                'category': WasteCategory.RECYCLABLE,
                'class': class_name,
                'confidence': confidence,
                'inferred': True,
                'instructions': 'Likely recyclable - check local guidelines',
                'color_code': 'blue',
                'icon': '♻️'
            }
        
        # Healthy keywords
        if any(keyword in class_name for keyword in ['fruit', 'vegetable', 'fresh', 'food', 'organic']):
            return {
                'category': WasteCategory.HEALTHY,
                'class': class_name,
                'confidence': confidence,
                'inferred': True,
                'instructions': 'Fresh food or compostable organic material',
                'color_code': 'green',
                'icon': '✓'
            }
        
        # Default to non-recyclable
        return {
            'category': WasteCategory.NON_RECYCLABLE,
            'class': class_name,
            'confidence': confidence,
            'inferred': True,
            'instructions': 'Unknown item - default to general waste',
            'color_code': 'gray',
            'icon': '🗑️'
        }
    
    def batch_categorize(self, detections: List[Dict]) -> Dict:
        """
        Categorize multiple detections and provide summary statistics
        
        Args:
            detections: List of detection dictionaries from model
            
        Returns:
            Summary with counts, recommendations, and environmental impact
        """
        category_counts = {
            WasteCategory.RECYCLABLE: 0,
            WasteCategory.NON_RECYCLABLE: 0,
            WasteCategory.HEALTHY: 0,
            WasteCategory.HAZARDOUS: 0
        }
        
        categorized_items = []
        hazardous_items = []
        recyclable_value = 0.0
        
        for detection in detections:
            class_name = detection.get('class', 'unknown')
            confidence = detection.get('confidence', 0.0)
            context = {
                'in_bin': detection.get('in_bin', False),
                'contaminated': detection.get('contaminated', False)
            }
            
            result = self.categorize(class_name, confidence, context)
            categorized_items.append(result)
            
            # Update counts
            category = result['category']
            category_counts[category] += 1
            
            # Track hazardous items
            if category == WasteCategory.HAZARDOUS:
                hazardous_items.append(result)
            
            # Estimate recycling value (placeholder)
            if category == WasteCategory.RECYCLABLE:
                recyclable_value += 0.05  # $0.05 per recyclable item
        
        # Generate summary
        total_items = len(detections)
        recycling_rate = (category_counts[WasteCategory.RECYCLABLE] / total_items * 100) if total_items > 0 else 0
        
        return {
            'total_items': total_items,
            'category_breakdown': {
                'recyclable': category_counts[WasteCategory.RECYCLABLE],
                'non_recyclable': category_counts[WasteCategory.NON_RECYCLABLE],
                'healthy': category_counts[WasteCategory.HEALTHY],
                'hazardous': category_counts[WasteCategory.HAZARDOUS]
            },
            'recycling_rate': round(recycling_rate, 1),
            'estimated_recycling_value': round(recyclable_value, 2),
            'hazardous_alert': len(hazardous_items) > 0,
            'hazardous_items': hazardous_items,
            'categorized_items': categorized_items,
            'recommendations': self._generate_recommendations(category_counts, total_items),
            'environmental_impact': self._calculate_environmental_impact(category_counts)
        }
    
    def _generate_recommendations(self, category_counts: Dict, total: int) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if category_counts[WasteCategory.HAZARDOUS] > 0:
            recommendations.append("⚠️ URGENT: Hazardous materials detected - arrange special disposal immediately")
        
        recyclable_pct = (category_counts[WasteCategory.RECYCLABLE] / total * 100) if total > 0 else 0
        
        if recyclable_pct < 30:
            recommendations.append("♻️ Low recycling rate - educate on recyclable materials")
        elif recyclable_pct > 70:
            recommendations.append("✓ Excellent recycling rate - keep up the good work!")
        
        if category_counts[WasteCategory.NON_RECYCLABLE] > total * 0.5:
            recommendations.append("⚠️ High non-recyclable waste - review packaging choices")
        
        if category_counts[WasteCategory.HEALTHY] > 0:
            recommendations.append("🌱 Organic waste detected - consider composting program")
        
        return recommendations
    
    def _calculate_environmental_impact(self, category_counts: Dict) -> Dict:
        """Calculate environmental impact metrics"""
        # Simplified impact calculation
        recyclable_impact = category_counts[WasteCategory.RECYCLABLE] * 2.5  # kg CO2 saved per item
        landfill_impact = category_counts[WasteCategory.NON_RECYCLABLE] * 0.5  # kg CO2 per item
        hazardous_impact = category_counts[WasteCategory.HAZARDOUS] * 10  # High impact
        
        return {
            'co2_saved_kg': round(recyclable_impact, 2),
            'co2_generated_kg': round(landfill_impact + hazardous_impact, 2),
            'net_impact': round(recyclable_impact - landfill_impact - hazardous_impact, 2),
            'impact_rating': self._get_impact_rating(recyclable_impact, landfill_impact + hazardous_impact)
        }
    
    def _get_impact_rating(self, saved: float, generated: float) -> str:
        """Get environmental impact rating"""
        net = saved - generated
        if net > 20:
            return "EXCELLENT - High positive impact"
        elif net > 10:
            return "GOOD - Positive impact"
        elif net > 0:
            return "FAIR - Slightly positive"
        else:
            return "POOR - Negative environmental impact"


# Singleton instance
_categorizer_instance = None

def get_waste_categorizer() -> WasteCategorizer:
    """Get singleton instance of waste categorizer"""
    global _categorizer_instance
    if _categorizer_instance is None:
        _categorizer_instance = WasteCategorizer()
        logger.info("Waste Categorizer initialized")
    return _categorizer_instance
