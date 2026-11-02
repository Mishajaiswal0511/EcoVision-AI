/**
 * EcoVision AI — Class Taxonomy and UI Theme Constants
 * Matches the trained YOLO11s model's 6 classes.
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
    examples: 'PET bottles, shampoo bottles, milk jugs, food tubs, clean film packaging',
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
    examples: 'Soda cans, food tins, aerosol cans, aluminium foil, metal caps',
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
    examples: 'Corrugated delivery boxes, cereal boxes, toilet paper rolls, packing sleeves',
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
    binName: 'Glass Bin / Dedicated Bank',
    binColorHex: '#0D9488',
    binBadge: 'bg-teal-600 text-white',
    examples: 'Beverage bottles, glass jars, cosmetic jars (handle carefully to avoid breakage)',
    iconName: 'Wine',
    recyclable: true,
    warning: 'Handle with care. If broken, wrap in newspaper before disposal.',
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
    examples: 'Newspapers, office paper, clean paper bags, notebooks, flyers (keep dry)',
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
    binName: 'Black / Grey Bin (General Waste)',
    binColorHex: '#4B5563',
    binBadge: 'bg-gray-700 text-white',
    examples: 'Greasy pizza boxes, soiled tissues, sanitary waste, mixed non-recyclables',
    iconName: 'Trash2',
    recyclable: false,
  },
];

export const CLASS_BY_NAME = Object.fromEntries(
  WASTE_CLASSES.map((c) => [c.name.toLowerCase(), c])
);

