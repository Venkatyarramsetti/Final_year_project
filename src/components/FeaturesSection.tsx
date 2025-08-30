import { Card, CardContent } from "@/components/ui/card";
import { Upload, Brain, Shield, Users, FileImage, AlertTriangle } from "lucide-react";

const FeaturesSection = () => {
  const features = [
    {
      icon: Upload,
      title: "Multi-Format Upload",
      description: "Support for JPG, PNG, PDF and other common file formats for comprehensive analysis.",
      color: "text-primary"
    },
    {
      icon: Brain,
      title: "AI Classification",
      description: "Advanced machine learning models trained specifically for hazard detection and safety assessment.",
      color: "text-accent"
    },
    {
      icon: Users,
      title: "Age-Specific Analysis",
      description: "Tailored safety assessments for teenagers, adults, and elderly individuals with specific risk factors.",
      color: "text-primary-glow"
    },
    {
      icon: Shield,
      title: "Safety Scoring",
      description: "Comprehensive safety scores with detailed explanations and recommended actions.",
      color: "text-primary-dark"
    },
    {
      icon: FileImage,
      title: "Batch Processing",
      description: "Upload and analyze multiple images simultaneously for efficient workflow management.",
      color: "text-accent"
    },
    {
      icon: AlertTriangle,
      title: "Risk Assessment",
      description: "Detailed risk categorization with severity levels and prevention recommendations.",
      color: "text-destructive"
    }
  ];

  return (
    <section id="features" className="py-20 bg-background">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <div className="inline-flex items-center px-4 py-2 bg-primary/10 rounded-full border border-primary/20 mb-6">
            <Brain className="w-4 h-4 text-primary mr-2" />
            <span className="text-sm font-medium text-primary">Advanced Features</span>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-foreground mb-6">
            Powerful AI Detection
          </h2>
          <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
            Our cutting-edge AI technology provides comprehensive hazard detection 
            with industry-leading accuracy and speed.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <Card 
              key={index} 
              className="group hover:shadow-primary transition-all duration-300 transform hover:-translate-y-2 bg-gradient-card border-border/50"
            >
              <CardContent className="p-8">
                <div className="mb-6">
                  <div className="w-16 h-16 bg-gradient-primary rounded-2xl flex items-center justify-center mb-4 group-hover:shadow-glow transition-all duration-300">
                    <feature.icon className={`w-8 h-8 text-primary-foreground`} />
                  </div>
                  <h3 className="text-xl font-bold text-foreground mb-3">{feature.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
                </div>
                <div className="h-1 bg-gradient-primary rounded-full transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300"></div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;