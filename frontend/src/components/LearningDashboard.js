import React from 'react';
import { 
    AlertTriangle
} from 'lucide-react';

const LearningDashboard = () => {
    // PATCH: Component crippled due to missing backend endpoints.
    return (
        <div className="flex items-center justify-center p-8 bg-slate-900/50 rounded-lg border border-yellow-800/30">
            <div className="text-center">
                <AlertTriangle className="w-12 h-12 text-yellow-600 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-slate-300">Learning Disabled</h3>
                <p className="text-slate-500 text-sm mt-2">Backend connection unavailable.</p>
            </div>
        </div>
    );
};

export default LearningDashboard;

/*
// LEGACY CODE BELOW - DISABLED - PRESERVED FOR REFERENCE
// 
// const LearningDashboard = () => {
//     const [summary, setSummary] = useState(null);
//     const [suggestions, setSuggestions] = useState({});
//     const [activeTab, setActiveTab] = useState('overview'); // 'overview', 'models', 'claims'
//     const [modelMetrics, setModelMetrics] = useState({});
//     const [claimTypes, setClaimTypes] = useState({});
//     const [loading, setLoading] = useState(true);

//     // Mock Data for "Learning" Simulation
//     const mockSummary = {
//         total_violations_recorded: 1243,
//         total_refusals_issued: 89,
//         models_tracked: 5,
//         user_satisfaction: 0.92,
//         feedback_collected: 450
//     };

//     useEffect(() => {
//         // Simulate fetching learning data
//         setTimeout(() => {
//             setSummary(mockSummary);
//             setLoading(false);
//         }, 1000);
//     }, []);

//     if (loading) {
//         return (
//             <div className="flex items-center justify-center h-64 text-slate-500">
//                 <Activity className="w-6 h-6 animate-spin mr-2" />
//                 Loading learning matrix...
//             </div>
//         );
//     }

//     if (!summary) return <div>No data available.</div>;

//     return (
//         <div className="flex-1 overflow-y-auto bg-slate-900/50 p-6 text-slate-200">
//             <div className="flex items-center justify-between mb-8">
//                 <h2 className="text-2xl font-bold flex items-center gap-3">
//                     <Shield className="w-6 h-6 text-emerald-400" />
//                     System Learning & Risk Analysis
//                 </h2>
//             </div>
//             <div className="text-center text-slate-500 mt-20">
//                 Legacy Dashboard View - Disabled
//             </div>
//         </div>
//     );
// };
*/
