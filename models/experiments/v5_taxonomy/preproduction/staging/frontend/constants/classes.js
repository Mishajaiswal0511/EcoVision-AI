/**
 * EcoVision AI — Class Taxonomy and UI Theme Constants (V5 7-Class Standard)
 * Aligned with Government of India Solid Waste Management Rules, 2026 (MoEFCC/CPCB)
 */

export const WASTE_CLASSES = [
  {
    id: 0,
    name: 'plastic',
    label: 'Plastic',
    color: '#2563EB', // Blue
    bgLight: 'bg-blue-50',
    borderLight: 'border-blue-200',
    badgeClass: 'bg-blue-100 text-blue-800 border-blue-200',
    binName: 'Blue Bin (Dry Recyclables)',
    binColorHex: '#2563EB',
    binBadge: 'bg-blue-500 text-white',
    examples: 'Clean PET beverage bottles, milk jugs, shampoo bottles, clean food tubs, clean plastic films',
    iconName: 'Package',
    recyclable: true,
  },
  {
    id: 1,
    name: 'metal',
    label: 'Metal',
    color: '#D97706', // Amber
    bgLight: 'bg-amber-50',
    borderLight: 'border-amber-200',
    badgeClass: 'bg-amber-100 text-amber-800 border-amber-200',
    binName: 'Blue Bin (Dry Recyclables)',
    binColorHex: '#D97706',
    binBadge: 'bg-amber-500 text-white',
    examples: 'Aluminium soda cans, tin/steel food cans, clean aluminium foil, metal caps & lids',
    iconName: 'ShieldAlert',
    recyclable: true,
  },
  {
    id: 2,
    name: 'cardboard',
    label: 'Cardboard',
    color: '#92400E', // Brown
    bgLight: 'bg-stone-50',
    borderLight: 'border-stone-300',
    badgeClass: 'bg-stone-100 text-stone-800 border-stone-300',
    binName: 'Blue Bin (Dry Recyclables)',
    binColorHex: '#92400E',
    binBadge: 'bg-amber-800 text-white',
    examples: 'Clean corrugated shipping boxes, dry cereal cartons, paperboard packaging (greasy pizza boxes go to General Trash)',
    iconName: 'Box',
    recyclable: true,
  },
  {
    id: 3,
    name: 'glass',
    label: 'Glass',
    color: '#0D9488', // Teal
    bgLight: 'bg-teal-50',
    borderLight: 'border-teal-200',
    badgeClass: 'bg-teal-100 text-teal-800 border-teal-200',
    binName: 'Blue Bin / Dedicated Glass Bank',
    binColorHex: '#0D9488',
    binBadge: 'bg-teal-600 text-white',
    examples: 'Container beverage bottles, glass food jars (rinse thoroughly; window glass & mirrors excluded)',
    iconName: 'Wine',
    recyclable: true,
    warning: '⚠️ SAFETY: If broken, wrap securely in multiple layers of newspaper before placing for collection.',
  },
  {
    id: 4,
    name: 'paper',
    label: 'Paper',
    color: '#059669', // Emerald
    bgLight: 'bg-emerald-50',
    borderLight: 'border-emerald-200',
    badgeClass: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    binName: 'Blue Bin (Dry Recyclables)',
    binColorHex: '#059669',
    binBadge: 'bg-emerald-600 text-white',
    examples: 'Office copy paper, clean paper bags, newspapers, magazines, notebooks (keep strictly dry)',
    iconName: 'FileText',
    recyclable: true,
  },
  {
    id: 5,
    name: 'general_trash',
    label: 'General Trash',
    color: '#E11D48', // Crimson
    bgLight: 'bg-rose-50',
    borderLight: 'border-rose-200',
    badgeClass: 'bg-rose-100 text-rose-800 border-rose-200',
    binName: 'Black / Grey Bin (Residual Waste)',
    binColorHex: '#4B5563',
    binBadge: 'bg-gray-700 text-white',
    examples: 'Multilayer snack/chip packets, Styrofoam clamshells, greasy pizza boxes, cigarette butts, non-recyclable composites',
    iconName: 'Trash2',
    recyclable: false,
  },
  {
    id: 6,
    name: 'organic',
    label: 'Organic Waste',
    color: '#16A34A', // Green
    bgLight: 'bg-green-50',
    borderLight: 'border-green-200',
    badgeClass: 'bg-green-100 text-green-800 border-green-200',
    binName: 'Green Bin (Wet Waste)',
    binColorHex: '#16A34A',
    binBadge: 'bg-green-600 text-white',
    examples: 'Fruit & vegetable peels, kitchen food leftovers, tea leaves, coffee grounds, eggshells, garden leaves',
    iconName: 'Apple',
    recyclable: false,
    compositable: true,
  },
];

export const CLASS_BY_NAME = Object.fromEntries(
  WASTE_CLASSES.map((cls) => [cls.name, cls])
);

export const CLASS_BY_ID = Object.fromEntries(
  WASTE_CLASSES.map((cls) => [cls.id, cls])
);

