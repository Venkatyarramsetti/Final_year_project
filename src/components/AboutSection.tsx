import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { CheckCircle, Target, Zap, Heart } from "lucide-react";

const AboutSection = () => {
  const benefits = [
    "Real-time hazard detection with 99.9% accuracy",
    "Age-specific safety recommendations",
    "Support for multiple file formats",
    "Comprehensive risk assessment reports",
    "Fast processing under 2 seconds",
    "Privacy-focused secure analysis"
  ];

  return (
    <section id="about" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Content */}
          <div>
            <div className="inline-flex items-center px-4 py-2 bg-accent/10 rounded-full border border-accent/20 mb-6">
              <Target className="w-4 h-4 text-accent mr-2" />
              <span className="text-sm font-medium text-accent">About HazardSpotter</span>
            </div>
            
            <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
              Protecting Lives Through
              <span className="text-primary"> AI Innovation</span>
            </h2>
            
            <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
              HazardSpotter leverages cutting-edge artificial intelligence to identify potential 
              safety hazards in images and documents. Our mission is to create safer environments 
              for everyone, with specialized focus on age-specific risk assessment.
            </p>
            
            <div className="space-y-4 mb-8">
              {benefits.map((benefit, index) => (
                <div key={index} className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-primary flex-shrink-0" />
                  <span className="text-foreground">{benefit}</span>
                </div>
              ))}
            </div>
            
            <Button 
              size="lg" 
              className="bg-gradient-primary shadow-primary hover:shadow-glow transition-all duration-300"
            >
              Learn More About Our Technology
            </Button>
          </div>
          
          {/* Cards */}
          <div className="space-y-6">
            <Card className="bg-gradient-card border-border/50 shadow-card">
              <CardContent className="p-8">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center mr-4">
                    <Zap className="w-6 h-6 text-primary" />
                  </div>
                  <h3 className="text-xl font-bold text-foreground">Lightning Fast</h3>
                </div>
                <p className="text-muted-foreground">
                  Process images in under 2 seconds with our optimized AI models 
                  running directly in your browser.
                </p>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-card border-border/50 shadow-card">
              <CardContent className="p-8">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mr-4">
                    <Heart className="w-6 h-6 text-accent" />
                  </div>
                  <h3 className="text-xl font-bold text-foreground">Privacy First</h3>
                </div>
                <p className="text-muted-foreground">
                  Your images are processed locally in your browser. No data is sent 
                  to external servers, ensuring complete privacy.
                </p>
              </CardContent>
            </Card>
            
            <Card className="bg-gradient-card border-border/50 shadow-card">
              <CardContent className="p-8">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-primary-glow/10 rounded-lg flex items-center justify-center mr-4">
                    <Target className="w-6 h-6 text-primary-glow" />
                  </div>
                  <h3 className="text-xl font-bold text-foreground">Accurate Detection</h3>
                </div>
                <p className="text-muted-foreground">
                  Our AI models are trained on millions of images to provide 
                  industry-leading accuracy in hazard detection.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;